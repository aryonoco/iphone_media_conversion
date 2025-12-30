#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
#
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025 Aryan Ameri
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
"""
Build LibRaw with Adobe DNG SDK support for iPhone 16/17 Pro JPEG-XL DNG files.

Works on Fedora, Debian/Ubuntu, Arch, and systems with Homebrew (Linux or macOS).

The Adobe DNG SDK is downloaded automatically from Adobe's servers.

Adobe DNG SDK: https://helpx.adobe.com/camera-raw/digital-negative.html
LibRaw: https://www.libraw.org/
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Self, override

if TYPE_CHECKING:
    pass

__all__: Final[list[str]] = [
    "BuildConfig",
    "Builder",
    "BuildError",
    "PackageManager",
    "main",
]

__version__: Final[str] = "1.1.0"

# Adobe DNG SDK download URL (redirects to actual zip file)
_DNG_SDK_DOWNLOAD_URL: Final[str] = "https://www.adobe.com/go/dng_sdk"

# Type aliases (PEP 695)
type CommandResult = subprocess.CompletedProcess[str]
type Environment = dict[str, str]


class BuildError(Exception):
    """Raised when a build step fails."""

    __slots__ = ("cmd",)

    def __init__(
        self,
        message: str,
        *,
        cmd: str | Sequence[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.cmd = cmd


class PackageManager(StrEnum):
    """Supported package managers."""

    HOMEBREW = auto()
    DNF = auto()
    APT = auto()
    PACMAN = auto()
    UNKNOWN = auto()


class _AnsiColor(StrEnum):
    """ANSI color codes for terminal output."""

    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    GRAY = "\033[0;37m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _get_cpu_count() -> int:
    """Get CPU count with fallback."""
    return os.cpu_count() or 1


def _get_env_path(var_name: str, /) -> Path | None:
    """Get a Path from environment variable, or None if not set/empty."""
    value = os.environ.get(var_name, "").strip()
    return Path(value) if value else None


def _get_env_int(var_name: str, /) -> int | None:
    """Get a positive int from environment variable, or None if invalid."""
    value = os.environ.get(var_name, "").strip()
    if not value:
        return None
    try:
        result = int(value)
        return result if result > 0 else None
    except ValueError:
        return None


def _get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).resolve().parent


@dataclass(slots=True, kw_only=True)
class BuildConfig:
    """Build configuration."""

    build_dir: Path
    install_prefix: Path
    jobs: int
    skip_deps: bool = False
    verbose: bool = False
    prefer_system_pkg: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.jobs < 1:
            msg = f"jobs must be >= 1, got {self.jobs}"
            raise ValueError(msg)
        if not self.build_dir.is_absolute():
            object.__setattr__(self, "build_dir", self.build_dir.resolve())
        if not self.install_prefix.is_absolute():
            object.__setattr__(self, "install_prefix", self.install_prefix.resolve())

    @property
    def deps_dir(self) -> Path:
        """Directory for built dependencies."""
        return self.build_dir / "deps"

    @classmethod
    def create(
        cls,
        *,
        build_dir: Path | None = None,
        install_prefix: Path | None = None,
        jobs: int | None = None,
        skip_deps: bool = False,
        verbose: bool = False,
        prefer_system_pkg: bool = False,
    ) -> Self:
        """Create config from arguments with environment variable fallbacks."""
        return cls(
            build_dir=(
                build_dir
                or _get_env_path("BUILD_DIR")
                or _get_script_dir() / "build"
            ),
            install_prefix=(
                install_prefix
                or _get_env_path("INSTALL_PREFIX")
                or _get_script_dir() / "build"
            ),
            jobs=jobs or _get_env_int("JOBS") or _get_cpu_count(),
            skip_deps=skip_deps or os.environ.get("SKIP_DEPS") == "1",
            verbose=verbose,
            prefer_system_pkg=prefer_system_pkg,
        )


@dataclass(slots=True, kw_only=True)
class Builder:
    """Main build orchestrator."""

    config: BuildConfig
    _logger: logging.Logger = field(init=False, repr=False)
    _brew_prefix_cache: Path | None = field(init=False, repr=False, default=None)
    _brew_prefix_checked: bool = field(init=False, repr=False, default=False)
    _dng_sdk_zip: Path | None = field(init=False, repr=False, default=None)
    _dng_sdk_dir: Path | None = field(init=False, repr=False, default=None)
    _dng_sdk_info: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        self._logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        """Configure colored logging."""
        logger = logging.getLogger(f"libraw-builder-{id(self)}")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_ColoredFormatter())
            logger.addHandler(handler)

        return logger

    @property
    def brew_prefix(self) -> Path | None:
        """Detect and cache Homebrew prefix."""
        if self._brew_prefix_checked:
            return self._brew_prefix_cache

        self._brew_prefix_checked = True

        try:
            result = subprocess.run(
                ["brew", "--prefix"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                prefix = result.stdout.strip()
                if prefix:
                    self._brew_prefix_cache = Path(prefix)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        return self._brew_prefix_cache

    def _build_env(self, extra_env: Environment | None = None) -> Environment:
        """Build environment dict with appropriate compiler and paths.

        With Homebrew (Linux/macOS): uses clang from llvm package with libc++.
        On systems with native package managers: uses system gcc/g++.
        """
        env = dict(os.environ)

        if extra_env:
            env |= extra_env

        prefix = self.brew_prefix
        is_linux = sys.platform.startswith("linux")
        is_macos = sys.platform == "darwin"

        if prefix is not None:
            # Homebrew environment - use clang from llvm on both Linux and macOS
            # This avoids GCC 15 + glibc issues on Linux and provides consistency
            llvm_bin = prefix / "opt" / "llvm" / "bin"
            env["CC"] = str(llvm_bin / "clang")
            env["CXX"] = str(llvm_bin / "clang++")

            # Add llvm bin to PATH for clang tools
            path_parts = [
                str(prefix / "opt" / "llvm" / "bin"),
                str(prefix / "bin"),
            ]
            current_path = env.get("PATH", "")
            if current_path:
                path_parts.append(current_path)
            env["PATH"] = ":".join(path_parts)

            pkg_paths = [
                f"{prefix}/lib/pkgconfig",
                f"{prefix}/opt/zlib/lib/pkgconfig",
                env.get("PKG_CONFIG_PATH", ""),
            ]
            env["PKG_CONFIG_PATH"] = ":".join(p for p in pkg_paths if p)

            # Use clang with libc++ on both Linux and macOS
            llvm_include = prefix / "opt" / "llvm" / "include"
            llvm_lib = prefix / "opt" / "llvm" / "lib"

            cpp_flags = [
                f"-I{llvm_include}/c++/v1",  # libc++ headers
                f"-I{prefix}/include",
                f"-I{prefix}/opt/zlib/include",
                env.get("CPPFLAGS", ""),
            ]
            env["CPPFLAGS"] = " ".join(p for p in cpp_flags if p)

            ld_flags = [
                f"-L{llvm_lib}",  # libc++ library
                f"-L{prefix}/lib",
                f"-L{prefix}/opt/zlib/lib",
                env.get("LDFLAGS", ""),
            ]
            env["LDFLAGS"] = " ".join(p for p in ld_flags if p)

            # Library path for finding libraries at link time
            lib_path = [
                f"{llvm_lib}",
                f"{prefix}/lib",
                env.get("LIBRARY_PATH", ""),
            ]
            env["LIBRARY_PATH"] = ":".join(p for p in lib_path if p)
        else:
            # System package managers: use standard gcc/g++
            env.setdefault("CC", "gcc")
            env.setdefault("CXX", "g++")

        return env

    def run(
        self,
        cmd: str | Sequence[str],
        /,
        *,
        cwd: Path | None = None,
        env: Environment | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Execute a command with merged environment."""
        full_env = self._build_env(env)
        shell = isinstance(cmd, str)

        self._logger.debug("Running: %s", cmd)

        # Normalize command for subprocess
        normalized_cmd: str | list[str]
        if shell:
            normalized_cmd = cmd
        elif isinstance(cmd, list):
            normalized_cmd = cmd
        else:
            normalized_cmd = list(cmd)

        # Always capture output so we can report errors
        result = subprocess.run(
            normalized_cmd,
            shell=shell,
            cwd=cwd,
            env=full_env,
            text=True,
            capture_output=True,
            check=False,
        )

        # In verbose mode, print stdout/stderr
        if self.config.verbose:
            if result.stdout:
                sys.stdout.write(result.stdout)
                sys.stdout.flush()
            if result.stderr:
                sys.stderr.write(result.stderr)
                sys.stderr.flush()

        if check and result.returncode != 0:
            error_output = result.stderr or result.stdout or "(no output)"
            self._logger.error("Command failed: %s\n%s", cmd, error_output)
            raise BuildError(
                f"Command failed with exit code {result.returncode}",
                cmd=cmd,
            )

        return result

    def detect_package_manager(self) -> PackageManager:
        """Detect available package manager, preferring Homebrew if available."""
        # Check for Homebrew first (works on Linux and macOS)
        if self.brew_prefix is not None and not self.config.prefer_system_pkg:
            return PackageManager.HOMEBREW

        # Fall back to system package manager
        if shutil.which("dnf"):
            return PackageManager.DNF
        if shutil.which("apt-get"):
            return PackageManager.APT
        if shutil.which("pacman"):
            return PackageManager.PACMAN

        # If prefer_system_pkg but no system manager found, try Homebrew
        if self.brew_prefix is not None:
            return PackageManager.HOMEBREW

        return PackageManager.UNKNOWN

    def install_deps(self) -> None:
        """Install build dependencies using detected package manager."""
        pkg_manager = self.detect_package_manager()
        self._logger.info("Using package manager: %s", pkg_manager)

        match pkg_manager:
            case PackageManager.HOMEBREW:
                self._install_homebrew_deps()
            case PackageManager.DNF:
                self._install_dnf_deps()
            case PackageManager.APT:
                self._install_apt_deps()
            case PackageManager.PACMAN:
                self._install_pacman_deps()
            case PackageManager.UNKNOWN:
                raise BuildError(
                    "No supported package manager found. "
                    "Install build deps manually and use --skip-deps"
                )

    def _install_homebrew_deps(self) -> None:
        """Install dependencies via Homebrew."""
        self._logger.info("Homebrew prefix: %s", self.brew_prefix)

        packages = [
            # Build tools - use llvm (clang with libc++) to avoid GCC 15 + glibc issues
            "cmake", "autoconf", "automake", "libtool", "pkg-config",
            "llvm",  # Provides clang/clang++ with libc++
            # Libraries
            "zlib", "jpeg-turbo", "little-cms2", "libpng", "brotli", "highway",
            # Utilities
            "git", "wget", "unzip",
        ]
        self.run(["brew", "install", *packages])

    def _install_dnf_deps(self) -> None:
        """Install dependencies via DNF (Fedora/RHEL)."""
        packages = [
            "gcc", "gcc-c++", "cmake", "make",
            "autoconf", "automake", "libtool", "pkg-config",
            "zlib-devel", "libjpeg-turbo-devel", "lcms2-devel", "libpng-devel",
            "brotli-devel", "highway-devel", "expat-devel", "libuuid-devel",
            "git", "unzip", "wget",
        ]
        self.run(["sudo", "dnf", "install", "-y", *packages])

    def _install_apt_deps(self) -> None:
        """Install dependencies via APT (Debian/Ubuntu)."""
        packages = [
            "build-essential", "cmake",
            "autoconf", "automake", "libtool", "pkg-config",
            "zlib1g-dev", "libjpeg-dev", "liblcms2-dev", "libpng-dev",
            "libbrotli-dev", "libhwy-dev", "libexpat1-dev", "uuid-dev",
            "git", "unzip", "wget",
        ]
        self.run(["sudo", "apt-get", "update"])
        self.run(["sudo", "apt-get", "install", "-y", *packages])

    def _install_pacman_deps(self) -> None:
        """Install dependencies via Pacman (Arch)."""
        packages = [
            "base-devel", "cmake",
            "autoconf", "automake", "libtool", "pkgconf",
            "zlib", "libjpeg-turbo", "lcms2", "libpng",
            "brotli", "highway", "expat",
            "git", "unzip", "wget",
        ]
        self.run(["sudo", "pacman", "-S", "--needed", "--noconfirm", *packages])

    def _download_dng_sdk(self) -> Path:
        """Download DNG SDK from Adobe, following redirect to get actual file.

        Downloads to build directory. Skips download if file already exists.
        Uses curl for reliable HTTP/2 and redirect handling.
        """
        self._logger.info("Setting up Adobe DNG SDK...")

        self.config.build_dir.mkdir(parents=True, exist_ok=True)

        try:
            # First, get the final URL after redirect using curl
            result = subprocess.run(
                ["curl", "-sI", "-L", "-o", "/dev/null", "-w", "%{url_effective}",
                 _DNG_SDK_DOWNLOAD_URL],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            final_url = result.stdout.strip()
            filename = Path(final_url).name

            if not filename.endswith(".zip"):
                raise BuildError(f"Unexpected file type from Adobe: {filename}")

            dest_path = self.config.build_dir / filename

            # Skip if already downloaded
            if dest_path.exists():
                self._logger.info("Using cached DNG SDK: %s", filename)
                return dest_path

            self._logger.info("Downloading %s...", filename)

            # Download with curl (handles HTTP/2, shows progress)
            result = subprocess.run(
                ["curl", "-L", "-o", str(dest_path), "--progress-bar",
                 _DNG_SDK_DOWNLOAD_URL],
                timeout=600,  # 10 min for ~80MB
                check=True,
            )

            self._logger.info("Downloaded: %s", dest_path.name)
            return dest_path

        except subprocess.CalledProcessError as exc:
            raise BuildError(
                f"Failed to download DNG SDK from Adobe (curl error)\n"
                f"URL: {_DNG_SDK_DOWNLOAD_URL}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BuildError(
                f"Download timed out after 10 minutes\n"
                f"URL: {_DNG_SDK_DOWNLOAD_URL}"
            ) from exc
        except FileNotFoundError:
            raise BuildError(
                "curl not found. Please install curl to download the DNG SDK."
            )

    def _extract_dng_sdk_info(self, zip_path: Path) -> str:
        """Extract version info from DNG SDK zip filename."""
        # Try to parse version from filename like: dng_sdk_1_7_1_2410_20251117.zip
        name = zip_path.stem  # Remove .zip
        
        # Try to extract version components
        # Pattern: dng_sdk_MAJOR_MINOR_PATCH_BUILD_DATE or dng_sdk_MAJOR_MINOR_PATCH
        match = re.match(
            r"dng_sdk_(\d+)_(\d+)_(\d+)(?:_(\d+))?(?:_(\d+))?",
            name,
        )
        
        if match:
            major, minor, patch, build, date = match.groups()
            version = f"{major}.{minor}.{patch}"
            if build:
                version += f" (build {build})"
            if date:
                # Format date if it looks like YYYYMMDD
                if len(date) == 8:
                    version += f", {date[:4]}-{date[4:6]}-{date[6:]}"
            return version
        
        # Fallback: just use the filename
        return name

    def _setup_dng_sdk(self) -> None:
        """Download and extract the DNG SDK."""
        # Download (or use cached)
        self._dng_sdk_zip = self._download_dng_sdk()
        self._dng_sdk_info = self._extract_dng_sdk_info(self._dng_sdk_zip)

        self._logger.info("DNG SDK version: %s", self._dng_sdk_info)

        # Check if already extracted (zip may extract to different name than zip stem)
        # e.g., dng_sdk_1_7_1_2410_20251117.zip extracts to dng_sdk_1_7_1/
        possible_dirs = list(self.config.build_dir.glob("dng_sdk*"))
        possible_dirs = [d for d in possible_dirs if d.is_dir()]

        if possible_dirs:
            # Use existing extraction
            self._dng_sdk_dir = sorted(possible_dirs)[-1]
            self._logger.info("Using existing DNG SDK: %s", self._dng_sdk_dir.name)
            return

        # Extract the zip
        self._logger.info("Extracting DNG SDK...")
        self.config.build_dir.mkdir(parents=True, exist_ok=True)

        with contextlib.chdir(self.config.build_dir):
            self.run(["unzip", "-q", str(self._dng_sdk_zip)])

        # Find the extracted directory
        possible_dirs = list(self.config.build_dir.glob("dng_sdk*"))
        possible_dirs = [d for d in possible_dirs if d.is_dir()]

        if possible_dirs:
            self._dng_sdk_dir = sorted(possible_dirs)[-1]
            self._logger.info("DNG SDK extracted to: %s", self._dng_sdk_dir.name)
        else:
            raise BuildError(
                f"Failed to find extracted DNG SDK directory in {self.config.build_dir}"
            )

    def download_sdks(self) -> None:
        """Set up DNG SDK and download required dependencies.

        Note: The DNG SDK bundles a minimal libjxl (library only, no tools).
        We clone the full libjxl from GitHub to get cjxl/djxl tools.
        """
        self.config.build_dir.mkdir(parents=True, exist_ok=True)

        # Set up DNG SDK (requires manual download from Adobe)
        self._setup_dng_sdk()

        # Download LibRaw
        libraw_dir = self.config.build_dir / "LibRaw"
        if not libraw_dir.exists():
            self._logger.info("Downloading LibRaw...")
            with contextlib.chdir(self.config.build_dir):
                self.run("git clone --depth 1 https://github.com/LibRaw/LibRaw.git")
        else:
            self._logger.info("Using existing LibRaw")

        # Download full libjxl (bundled version is minimal, missing tools)
        libjxl_dir = self.config.build_dir / "libjxl"
        if not libjxl_dir.exists():
            self._logger.info("Downloading libjxl...")
            with contextlib.chdir(self.config.build_dir):
                self.run(
                    "git clone --depth 1 --recurse-submodules --shallow-submodules "
                    "https://github.com/libjxl/libjxl.git"
                )
        else:
            self._logger.info("Using existing libjxl")

    def build_libjxl(self) -> None:
        """Build JPEG-XL library from GitHub clone.

        Uses bundled highway and brotli from libjxl's third_party directory.
        Also builds cjxl/djxl tools for encoding/decoding.
        """
        self._logger.info("Building libjxl...")
        src = self.config.build_dir / "libjxl"

        if not src.exists():
            raise BuildError(f"libjxl not found at {src}. Call download_sdks() first.")

        # Build flags for cmake (needed to find headers and libraries)
        # Use clang with libc++ for consistency across platforms
        cxx_flags_parts: list[str] = []
        linker_flags_parts: list[str] = []
        prefix = self.brew_prefix
        if prefix is not None:
            llvm_include = prefix / "opt" / "llvm" / "include"
            llvm_lib = prefix / "opt" / "llvm" / "lib"
            # Use libc++ (clang's native C++ standard library)
            cxx_flags_parts.extend([
                "-stdlib=libc++",
                f"-I{llvm_include}/c++/v1",
                f"-I{prefix}/include",
            ])
            linker_flags_parts.extend([
                "-stdlib=libc++",
                f"-L{llvm_lib}",
                f"-L{prefix}/lib",
                f"-L{prefix}/opt/zlib/lib",
            ])
        cxx_flags = " ".join(cxx_flags_parts)
        linker_flags = " ".join(linker_flags_parts)

        # Combine include and linker flags for CMAKE_CXX_FLAGS
        all_cxx_flags = f"{cxx_flags} {linker_flags}".strip()

        cmake_args = [
            "cmake", "-B", "build",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={self.config.deps_dir}",
            # Pass flags via CXX_FLAGS so cmake's try_compile tests inherit them
            f"-DCMAKE_CXX_FLAGS={all_cxx_flags}",
            f"-DCMAKE_EXE_LINKER_FLAGS={linker_flags}",
            f"-DCMAKE_SHARED_LINKER_FLAGS={linker_flags}",
            "-DBUILD_TESTING=OFF",
            # Build cjxl/djxl tools for encoding/decoding
            "-DJPEGXL_ENABLE_TOOLS=ON",
            "-DJPEGXL_ENABLE_EXAMPLES=OFF",
            "-DJPEGXL_ENABLE_SJPEG=OFF",
            "-DJPEGXL_ENABLE_JPEGLI=OFF",
            "-DJPEGXL_ENABLE_BENCHMARK=OFF",
            "-DJPEGXL_ENABLE_DOXYGEN=OFF",
            "-DJPEGXL_ENABLE_MANPAGES=OFF",
            # Use bundled highway and brotli from third_party/
            "-DJPEGXL_FORCE_SYSTEM_BROTLI=OFF",
            "-DJPEGXL_FORCE_SYSTEM_HWY=OFF",
        ]

        with contextlib.chdir(src):
            self.run(cmake_args)
            self.run(["cmake", "--build", "build", f"-j{self.config.jobs}"])
            self.run(["cmake", "--install", "build"])

    def build_xmp(self) -> None:
        """Build complete XMP SDK (XMPCore + XMPFiles + utilities).

        The DNG SDK includes a complete XMP toolkit with bundled expat.
        XMP is required for JXL support (dng_jxl.cpp uses XMP unconditionally).
        This builds XMPCore, XMPFiles, XMPCommon, MD5, and all utilities.
        """
        if self._dng_sdk_dir is None:
            raise BuildError("DNG SDK not set up. Call download_sdks() first.")

        self._logger.info("Building complete XMP SDK...")

        xmp_toolkit = self._dng_sdk_dir / "xmp" / "toolkit"
        if not xmp_toolkit.exists():
            raise BuildError(f"XMP toolkit not found at {xmp_toolkit}")

        # Create build directory for XMP
        xmp_build_dir = self.config.build_dir / "xmp_build"
        xmp_build_dir.mkdir(parents=True, exist_ok=True)

        # Key paths
        expat_path = xmp_toolkit / "XMPCore" / "third-party" / "expat" / "public" / "lib"
        public_include = xmp_toolkit / "public" / "include"

        # Determine platform-specific defines and exclusions
        is_linux = sys.platform.startswith("linux")
        is_macos = sys.platform == "darwin"

        if is_linux:
            platform_defines = "-DqLinux=1 -DUNIX_ENV=1 -DXMP_UNIXBuild=1"
            # Exclude non-Linux platform files
            platform_excludes = """
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*_Win\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*_Mac\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*_Android\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*OS_Utils_Mac\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*OS_Utils_WIN\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*OS_Utils_Android\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*Host_IO-Win\\\\.cpp$")
"""
        elif is_macos:
            platform_defines = "-DqMacOS=1 -DMAC_ENV=1 -DXMP_MacBuild=1"
            platform_excludes = """
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*_Win\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*_Linux\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*_Android\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*OS_Utils_WIN\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*OS_Utils_Linux\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*OS_Utils_Android\\\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*Host_IO-Win\\\\.cpp$")
"""
        else:
            platform_defines = "-DUNIX_ENV=1"
            platform_excludes = ""

        # Build flags for Homebrew (clang/libc++)
        cxx_flags = ""
        linker_flags = ""
        prefix = self.brew_prefix
        if prefix is not None:
            llvm_include = prefix / "opt" / "llvm" / "include"
            llvm_lib = prefix / "opt" / "llvm" / "lib"
            cxx_flags = f"-stdlib=libc++ -I{llvm_include}/c++/v1"
            linker_flags = f"-stdlib=libc++ -L{llvm_lib} -L{prefix}/lib"

        # Create CMakeLists.txt for complete XMP SDK
        cmake_content = f"""\
cmake_minimum_required(VERSION 3.16)
project(XMP C CXX)

# Use C++14 because bundled Boost uses std::auto_ptr (removed in C++17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Platform and XMP configuration defines
add_definitions(
    {platform_defines}
    -DkBigEndianHost=0
    -DXMP_StaticBuild=1
    -DXML_STATIC=1
    -DXML_DTD=1
    -DXML_NS=1
    -DXML_CONTEXT_BYTES=1024
    -DENABLE_CPP_DOM_MODEL=0
    -DXMP_COMPONENT_INT_NAMESPACE=AdobeXMPCore_Int
    -DXMP_COMPONENT_INT_NAMESPACE_V2=AdobeXMPCoreInt
    -DBUILDING_XMPCORE_LIB=1
    -DBUILDING_XMPCORE_AS_STATIC=1
    # Enable private handlers (needed for HEIF, WebP, CanonXF)
    -DAdobePrivate=1
)

# Expat sources (bundled)
set(EXPAT_SOURCES
    {expat_path}/xmlparse.c
    {expat_path}/xmlrole.c
    {expat_path}/xmltok.c
)

# XMPCore sources
file(GLOB XMPCORE_SOURCES "{xmp_toolkit}/XMPCore/source/*.cpp")

# XMPFiles sources (main + subdirectories)
file(GLOB XMPFILES_SOURCES "{xmp_toolkit}/XMPFiles/source/*.cpp")
file(GLOB XMPFILES_HANDLERS "{xmp_toolkit}/XMPFiles/source/FileHandlers/*.cpp")
file(GLOB XMPFILES_FORMAT "{xmp_toolkit}/XMPFiles/source/FormatSupport/*.cpp")
file(GLOB XMPFILES_AIFF "{xmp_toolkit}/XMPFiles/source/FormatSupport/AIFF/*.cpp")
file(GLOB XMPFILES_IFF "{xmp_toolkit}/XMPFiles/source/FormatSupport/IFF/*.cpp")
file(GLOB XMPFILES_WAVE "{xmp_toolkit}/XMPFiles/source/FormatSupport/WAVE/*.cpp")
file(GLOB XMPFILES_WEBP "{xmp_toolkit}/XMPFiles/source/FormatSupport/WebP/*.cpp")
file(GLOB XMPFILES_NATIVE "{xmp_toolkit}/XMPFiles/source/NativeMetadataSupport/*.cpp")
file(GLOB XMPFILES_PLUGIN "{xmp_toolkit}/XMPFiles/source/PluginHandler/*.cpp")

# XMPCommon sources
file(GLOB XMPCOMMON_SOURCES "{xmp_toolkit}/XMPCommon/source/*.cpp")

# Main toolkit utility sources
file(GLOB XMP_UTILITY_SOURCES "{xmp_toolkit}/source/*.cpp")

# MD5 and UUID sources (third-party/zuid)
file(GLOB ZUID_SOURCES "{xmp_toolkit}/third-party/zuid/sources/*.cpp")

# Combine all sources
set(ALL_SOURCES
    ${{EXPAT_SOURCES}}
    ${{XMPCORE_SOURCES}}
    ${{XMPFILES_SOURCES}}
    ${{XMPFILES_HANDLERS}}
    ${{XMPFILES_FORMAT}}
    ${{XMPFILES_AIFF}}
    ${{XMPFILES_IFF}}
    ${{XMPFILES_WAVE}}
    ${{XMPFILES_WEBP}}
    ${{XMPFILES_NATIVE}}
    ${{XMPFILES_PLUGIN}}
    ${{XMPCOMMON_SOURCES}}
    ${{XMP_UTILITY_SOURCES}}
    ${{ZUID_SOURCES}}
)

# Exclude platform-specific files not for this platform
{platform_excludes}
# Exclude Android-specific files
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*Android_Utils\\\\.cpp$")

add_library(XMPCore STATIC ${{ALL_SOURCES}})

target_include_directories(XMPCore PUBLIC
    {public_include}
    {xmp_toolkit}
    {xmp_toolkit}/XMPCore/source
    {xmp_toolkit}/XMPCore/third-party/boost
    {xmp_toolkit}/XMPFiles/source
    {xmp_toolkit}/XMPCommon/source
    {xmp_toolkit}/XMPFilesPlugins/api/source
    {xmp_toolkit}/source
    {xmp_toolkit}/third-party/zuid/interfaces
    {expat_path}
)

install(TARGETS XMPCore ARCHIVE DESTINATION lib)
install(DIRECTORY {public_include}/ DESTINATION include/xmp
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.incl_cpp")
"""
        cmake_file = xmp_build_dir / "CMakeLists.txt"
        cmake_file.write_text(cmake_content, encoding="utf-8")

        # Configure and build
        cmake_args = [
            "cmake", "-B", "build",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={self.config.deps_dir}",
        ]
        # Add -fPIC for position-independent code (required for shared library linking)
        pic_flag = "-fPIC"
        if cxx_flags:
            cmake_args.append(f"-DCMAKE_CXX_FLAGS={pic_flag} {cxx_flags}")
            cmake_args.append(f"-DCMAKE_C_FLAGS={pic_flag} -I{prefix}/opt/llvm/include" if prefix else f"-DCMAKE_C_FLAGS={pic_flag}")
        else:
            cmake_args.append(f"-DCMAKE_CXX_FLAGS={pic_flag}")
            cmake_args.append(f"-DCMAKE_C_FLAGS={pic_flag}")
        if linker_flags:
            cmake_args.extend([
                f"-DCMAKE_EXE_LINKER_FLAGS={linker_flags}",
                f"-DCMAKE_SHARED_LINKER_FLAGS={linker_flags}",
            ])
        # Remove empty args
        cmake_args = [arg for arg in cmake_args if arg]

        with contextlib.chdir(xmp_build_dir):
            self.run(cmake_args)
            self.run(["cmake", "--build", "build", f"-j{self.config.jobs}"])
            self.run(["cmake", "--install", "build"])

    def build_dng_sdk(self) -> None:
        """Build Adobe DNG SDK with libjxl and XMP support."""
        if self._dng_sdk_dir is None:
            raise BuildError("DNG SDK not set up. Call download_sdks() first.")

        self._logger.info("Building DNG SDK with XMP support...")
        src = self._dng_sdk_dir / "dng_sdk"

        if not src.exists():
            raise BuildError(f"DNG SDK source not found at {src}")

        # Use installed libjxl and XMPCore from deps directory
        deps_include = self.config.deps_dir / "include"
        deps_lib = self.config.deps_dir / "lib"

        # XMP toolkit paths (bundled with DNG SDK)
        xmp_toolkit = self._dng_sdk_dir / "xmp" / "toolkit"
        xmp_public_include = xmp_toolkit / "public" / "include"

        # Build additional paths for Homebrew (using llvm/libc++)
        extra_include_dirs = ""
        extra_lib_dirs = ""
        cxx_flags = ""
        linker_flags = ""
        prefix = self.brew_prefix
        if prefix is not None:
            llvm_include = prefix / "opt" / "llvm" / "include"
            llvm_lib = prefix / "opt" / "llvm" / "lib"
            jpeg_include = prefix / "opt" / "jpeg-turbo" / "include"
            extra_include_dirs = f"\n    {jpeg_include}"
            extra_lib_dirs = f"\n    {llvm_lib}\n    {prefix}/lib"
            cxx_flags = f"-stdlib=libc++ -I{llvm_include}/c++/v1"
            linker_flags = f"-stdlib=libc++ -L{llvm_lib} -L{prefix}/lib"

        # Determine platform define
        is_linux = sys.platform.startswith("linux")
        platform_define = "-DqLinux=1" if is_linux else "-DqMacOS=1"

        cmake_content = f"""\
cmake_minimum_required(VERSION 3.16)
project(dng_sdk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform and DNG SDK configuration defines
add_definitions(
    {platform_define}
    -DqDNGThreadSafe=1
    -DqDNGUseLibJPEG=1
    -DqDNGUseXMP=1
    -DqDNGValidate=0
    -DXMP_StaticBuild=1
    # XMP environment defines (required for XMP headers)
    -DUNIX_ENV=1
    -DXMP_UNIXBuild=1
)

file(GLOB DNG_SOURCES "source/*.cpp")
list(FILTER DNG_SOURCES EXCLUDE REGEX ".*dng_validate\\\\.cpp$")

add_library(dng STATIC ${{DNG_SOURCES}})

target_include_directories(dng PUBLIC
    source
    {deps_include}
    {deps_include}/xmp
    {xmp_public_include}
    {xmp_toolkit}{extra_include_dirs}
)

# Link libjxl, XMPCore, and other dependencies
target_link_directories(dng PUBLIC
    {deps_lib}{extra_lib_dirs}
)
target_link_libraries(dng PUBLIC
    XMPCore
    jxl jxl_threads hwy
    brotlienc brotlidec brotlicommon
    jpeg pthread z m
)

install(TARGETS dng ARCHIVE DESTINATION lib)
install(DIRECTORY source/ DESTINATION include/dng_sdk FILES_MATCHING PATTERN "*.h")
"""
        (src / "CMakeLists.txt").write_text(cmake_content, encoding="utf-8")

        cmake_args = [
            "cmake", "-B", "build",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={self.config.deps_dir}",
        ]
        # Add -fPIC for position-independent code (required for shared library linking)
        pic_flag = "-fPIC"
        if cxx_flags:
            cmake_args.append(f"-DCMAKE_CXX_FLAGS={pic_flag} {cxx_flags}")
        else:
            cmake_args.append(f"-DCMAKE_CXX_FLAGS={pic_flag}")
        if linker_flags:
            cmake_args.extend([
                f"-DCMAKE_EXE_LINKER_FLAGS={linker_flags}",
                f"-DCMAKE_SHARED_LINKER_FLAGS={linker_flags}",
            ])

        with contextlib.chdir(src):
            self.run(cmake_args)
            self.run(["cmake", "--build", "build", f"-j{self.config.jobs}"])
            self.run(["cmake", "--install", "build"])

    def build_libraw(self) -> None:
        """Build LibRaw with DNG SDK support."""
        if self._dng_sdk_dir is None:
            raise BuildError("DNG SDK not set up. Call download_sdks() first.")

        self._logger.info("Building LibRaw with DNG SDK support...")
        src = self.config.build_dir / "LibRaw"

        if not src.exists():
            raise BuildError(f"LibRaw source not found at {src}")

        # Build environment - use installed dependencies from deps directory
        include_dirs: list[Path] = [
            self.config.deps_dir / "include",
            self.config.deps_dir / "include" / "dng_sdk",
        ]
        lib_dirs: list[Path] = [self.config.deps_dir / "lib"]

        # Add Homebrew paths if available (using llvm/libc++)
        prefix = self.brew_prefix
        if prefix is not None:
            include_dirs.extend([
                prefix / "opt" / "llvm" / "include" / "c++" / "v1",  # libc++ headers
                prefix / "include",
                prefix / "opt" / "jpeg-turbo" / "include",
                prefix / "opt" / "zlib" / "include",
            ])
            lib_dirs.extend([
                prefix / "opt" / "llvm" / "lib",  # libc++ library
                prefix / "lib",
                prefix / "opt" / "jpeg-turbo" / "lib",
                prefix / "opt" / "zlib" / "lib",
            ])

        # Build with libc++ when using Homebrew (clang)
        # Need to define qLinux for DNG SDK headers and UNIX_ENV/XMP_UNIXBuild for XMP
        is_linux = sys.platform.startswith("linux")
        platform_defines = "-DqLinux=1 -DUNIX_ENV=1 -DXMP_UNIXBuild=1" if is_linux else "-DqMacOS=1 -DMAC_ENV=1"
        cxxflags = f"-O2 -DUSE_DNGSDK=1 {platform_defines}"
        libs = (
            "-ldng -lXMPCore -ljxl -ljxl_threads -lhwy "
            "-lbrotlienc -lbrotlidec -lbrotlicommon "
            "-ljpeg -lz -lpthread -lm"
        )
        if prefix is not None:
            cxxflags = f"-stdlib=libc++ {cxxflags}"
            # Add libc++ and OpenMP from LLVM
            libs = f"-lc++ -lomp {libs}"

        # Build LD_LIBRARY_PATH for runtime library search (needed for configure tests)
        ld_lib_paths = [str(d) for d in lib_dirs]
        if os.environ.get("LD_LIBRARY_PATH"):
            ld_lib_paths.append(os.environ["LD_LIBRARY_PATH"])

        env: Environment = {
            "CPPFLAGS": " ".join(f"-I{d}" for d in include_dirs),
            "LDFLAGS": " ".join(f"-L{d}" for d in lib_dirs),
            "LIBS": libs,
            "PKG_CONFIG_PATH": ":".join(
                p for p in [
                    str(self.config.deps_dir / "lib" / "pkgconfig"),
                    os.environ.get("PKG_CONFIG_PATH", ""),
                ] if p
            ),
            "CXXFLAGS": cxxflags,
            "LD_LIBRARY_PATH": ":".join(ld_lib_paths),
        }

        with contextlib.chdir(src):
            self.run(["autoreconf", "-fiv"])
            self.run([
                "./configure",
                f"--prefix={self.config.install_prefix}",
                "--enable-static",
                "--disable-examples",
                "--enable-jpeg",
                "--enable-lcms",
            ], env=env)
            self.run(["make", f"-j{self.config.jobs}"], env=env)
            self.run(["make", "install"], env=env)

        self._build_dcraw_emu(src, include_dirs, lib_dirs)

    def _build_dcraw_emu(
        self,
        src: Path,
        include_dirs: list[Path],
        lib_dirs: list[Path],
    ) -> None:
        """Build dcraw_emu binary with DNG SDK support.

        The binary is placed in the build directory (install_prefix).
        """
        self._logger.info("Building dcraw_emu with DNG SDK support...")

        # Output to build directory
        output_binary = self.config.install_prefix / "dcraw_emu_dng"

        # Get compiler from build environment (clang for Homebrew, g++ for system)
        build_env = self._build_env()
        cxx = build_env.get("CXX", "g++")

        # Build paths
        all_includes: list[str] = [
            ".",
            "./libraw",
            *(str(p) for p in include_dirs),
        ]
        all_libs: list[str] = [
            "./lib/.libs",
            *(str(p) for p in lib_dirs),
        ]

        # Choose C++ library based on compiler (libc++ for clang, libstdc++ for gcc)
        # Also add OpenMP library for LibRaw's parallel processing
        cxx_libs: list[str] = []
        if self.brew_prefix is not None:
            # Using clang with libc++ and OpenMP from LLVM
            cxx_stdlib_flag = "-stdlib=libc++"
            cxx_libs = ["-lc++", "-lomp"]
            all_libs.append(f"{self.brew_prefix}/opt/llvm/lib")
        else:
            # Using system gcc with libstdc++ and OpenMP
            cxx_stdlib_flag = ""
            cxx_libs = ["-lstdc++", "-lgomp"]

        # Platform defines needed for DNG SDK headers
        is_linux = sys.platform.startswith("linux")
        platform_defines = ["-DqLinux=1", "-DUNIX_ENV=1", "-DXMP_UNIXBuild=1"] if is_linux else ["-DqMacOS=1", "-DMAC_ENV=1"]

        compile_cmd: list[str] = [
            cxx, "-O2", "-DUSE_DNGSDK=1", *platform_defines,
        ]
        if cxx_stdlib_flag:
            compile_cmd.append(cxx_stdlib_flag)
        compile_cmd.extend([
            *(f"-I{p}" for p in all_includes),
            "samples/dcraw_emu.cpp",
            *(f"-L{p}" for p in all_libs),
            "-lraw", "-ldng", "-lXMPCore", "-ljxl", "-ljxl_threads", "-lhwy",
            "-lbrotlienc", "-lbrotlidec", "-lbrotlicommon",
            "-ljpeg", "-llcms2", "-lz", "-lpthread", "-lm", *cxx_libs,
            "-o", str(output_binary),
        ])

        with contextlib.chdir(src):
            self.run(compile_cmd)

    def verify(self) -> None:
        """Verify the build succeeded."""
        binary = self.config.install_prefix / "dcraw_emu_dng"

        if not binary.exists():
            raise BuildError(f"Build failed - binary not found at {binary}")

        if not os.access(binary, os.X_OK):
            raise BuildError(f"Build failed - binary not executable: {binary}")

        # Get binary info
        self.run([str(binary)], check=False)

        c = _AnsiColor
        print(f"""
{c.GREEN}{'=' * 50}{c.RESET}
{c.GREEN}{c.BOLD}Build successful!{c.RESET}
{c.GREEN}{'=' * 50}{c.RESET}

{c.BOLD}Binary:{c.RESET} {binary}
{c.BOLD}DNG SDK:{c.RESET} {self._dng_sdk_info}

{c.BOLD}Usage for iPhone 16/17 Pro DNG files:{c.RESET}
  {binary} -dngsdk -T -Z output.tiff input.DNG

{c.GRAY}All build artifacts are in: {self.config.build_dir}
Use export_photos.py to process iPhone ProRAW files.{c.RESET}
""")

    def build_all(self) -> None:
        """Execute full build pipeline."""
        if not self.config.skip_deps:
            self.install_deps()

        # Log compiler choice
        build_env = self._build_env()
        cc = build_env.get("CC", "cc")
        cxx = build_env.get("CXX", "c++")
        self._logger.info("Using compilers: CC=%s, CXX=%s", cc, cxx)

        self.download_sdks()
        self.build_libjxl()
        self.build_xmp()
        self.build_dng_sdk()
        self.build_libraw()
        self.verify()


class _ColoredFormatter(logging.Formatter):
    """Logging formatter with colored output."""

    _LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: _AnsiColor.GRAY,
        logging.INFO: _AnsiColor.GREEN,
        logging.WARNING: _AnsiColor.YELLOW,
        logging.ERROR: _AnsiColor.RED,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, _AnsiColor.RESET)
        return f"{color}[{record.levelname}]{_AnsiColor.RESET} {record.getMessage()}"


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    default_jobs = _get_cpu_count()

    parser = argparse.ArgumentParser(
        description=(
            "Build LibRaw with Adobe DNG SDK support "
            "for iPhone 16/17 Pro JPEG-XL DNG files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
The Adobe DNG SDK is downloaded automatically from Adobe's servers.

All build artifacts are placed in ./build/ by default.

Environment variables:
  BUILD_DIR        Build directory (default: ./build)
  INSTALL_PREFIX   Installation prefix (default: ./build)
  JOBS             Parallel build jobs (default: {default_jobs})
  SKIP_DEPS        Set to "1" to skip dependency installation

Examples:
  %(prog)s                           # Build with defaults
  %(prog)s -j 8 -v                   # 8 parallel jobs, verbose
  %(prog)s --skip-deps               # Skip package installation
  %(prog)s --prefer-system-pkg       # Use apt/dnf/pacman over Homebrew
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Build directory (default: ./build)",
    )
    parser.add_argument(
        "--install-prefix",
        type=Path,
        default=None,
        metavar="DIR",
        help="Installation prefix (default: ./build)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        metavar="N",
        help=f"Parallel build jobs (default: {default_jobs})",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip system dependency installation",
    )
    parser.add_argument(
        "--prefer-system-pkg",
        action="store_true",
        help="Prefer system package manager over Homebrew",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    try:
        config = BuildConfig.create(
            build_dir=args.build_dir,
            install_prefix=args.install_prefix,
            jobs=args.jobs,
            skip_deps=args.skip_deps,
            verbose=args.verbose,
            prefer_system_pkg=args.prefer_system_pkg,
        )
    except ValueError as exc:
        sys.exit(f"Error: Invalid configuration: {exc}")

    c = _AnsiColor
    print(f"""
{c.BOLD}{'=' * 50}
LibRaw + DNG SDK Builder v{__version__}
For iPhone 16/17 Pro JPEG-XL DNG support
{'=' * 50}{c.RESET}

Build directory: {config.build_dir}
Install prefix:  {config.install_prefix}
Parallel jobs:   {config.jobs}
""")

    try:
        builder = Builder(config=config)
        builder.build_all()
    except BuildError as exc:
        print(f"\n{c.RED}[ERROR]{c.RESET} {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n{c.YELLOW}Build interrupted by user.{c.RESET}", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
