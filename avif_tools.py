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
"""
Build AVIF tools (avifenc, avifdec, avifgainmaputil) from source.

This script builds libavif with AVIF_BUILD_APPS=ON to produce command-line
tools for AVIF encoding, decoding, and HDR gain map manipulation.

All dependencies (aom, zlib, libpng, libjpeg-turbo) come from system packages.
Only libavif is built from source to get avifgainmaputil.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Final, Self, override

__all__: Final[list[str]] = [
    "BuildConfig",
    "Builder",
    "BuildError",
    "PackageManager",
    "main",
]

__version__: Final[str] = "1.0.0"

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
    ZYPPER = auto()
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


def _is_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0


def _check_sudo_available() -> bool:
    """Check if sudo is available."""
    return shutil.which("sudo") is not None


def _privileged_cmd(cmd: list[str]) -> list[str]:
    """Wrap command with sudo if not running as root."""
    if _is_root():
        return cmd
    if not _check_sudo_available():
        raise BuildError(
            "This script requires root privileges to install system packages.\n"
            "Please run with sudo or as root:\n"
            f"  sudo {sys.executable} {' '.join(sys.argv)}\n"
            "Or install dependencies manually and use --skip-deps"
        )
    return ["sudo", *cmd]


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
    def sources_dir(self) -> Path:
        """Directory for source code."""
        return self.build_dir / "sources"

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
                or _get_script_dir() / "avif_build"
            ),
            install_prefix=(
                install_prefix
                or _get_env_path("INSTALL_PREFIX")
                or _get_script_dir() / "avif_build"
            ),
            jobs=jobs or _get_env_int("JOBS") or _get_cpu_count(),
            skip_deps=skip_deps or os.environ.get("SKIP_DEPS") == "1",
            verbose=verbose,
            prefer_system_pkg=prefer_system_pkg,
        )


@dataclass(slots=True, kw_only=True)
class Builder:
    """Main build orchestrator for AVIF tools."""

    config: BuildConfig
    _logger: logging.Logger = field(init=False, repr=False)
    _brew_prefix_cache: Path | None = field(init=False, repr=False, default=None)
    _brew_prefix_checked: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        """Configure colored logging."""
        logger = logging.getLogger(f"avif-builder-{id(self)}")
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

        Always uses clang with libc++ to avoid GCC 14+ compatibility issues.
        """
        env = dict(os.environ)

        if extra_env:
            env |= extra_env

        prefix = self.brew_prefix
        is_linux = sys.platform.startswith("linux")

        if prefix is not None:
            # Homebrew environment - use clang from llvm
            llvm_bin = prefix / "opt" / "llvm" / "bin"
            env["CC"] = str(llvm_bin / "clang")
            env["CXX"] = str(llvm_bin / "clang++")

            # Add llvm bin to PATH
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
                f"{prefix}/opt/jpeg-turbo/lib/pkgconfig",
                env.get("PKG_CONFIG_PATH", ""),
            ]
            env["PKG_CONFIG_PATH"] = ":".join(p for p in pkg_paths if p)

            # libc++ flags
            llvm_include = prefix / "opt" / "llvm" / "include"
            llvm_lib = prefix / "opt" / "llvm" / "lib"

            cpp_flags = [
                f"-I{llvm_include}/c++/v1",
                f"-I{prefix}/include",
                f"-I{prefix}/opt/jpeg-turbo/include",
                env.get("CPPFLAGS", ""),
            ]
            env["CPPFLAGS"] = " ".join(p for p in cpp_flags if p)

            ld_flags = [
                f"-L{llvm_lib}",
                f"-L{prefix}/lib",
                f"-L{prefix}/opt/jpeg-turbo/lib",
                env.get("LDFLAGS", ""),
            ]
            env["LDFLAGS"] = " ".join(p for p in ld_flags if p)

        else:
            # System package managers: use clang with libc++
            env.setdefault("CC", "clang")
            env.setdefault("CXX", "clang++")
            existing_cxxflags = env.get("CXXFLAGS", "")
            existing_ldflags = env.get("LDFLAGS", "")
            env["CXXFLAGS"] = f"-stdlib=libc++ {existing_cxxflags}".strip()
            env["LDFLAGS"] = f"-stdlib=libc++ {existing_ldflags}".strip()

        return env

    def _cmake_compiler_flags(self) -> list[str]:
        """Generate CMake compiler flags for clang + libc++."""
        flags = []

        cxx_flags = "-stdlib=libc++"
        linker_flags = "-stdlib=libc++"

        prefix = self.brew_prefix
        if prefix is not None:
            llvm_include = prefix / "opt" / "llvm" / "include"
            llvm_lib = prefix / "opt" / "llvm" / "lib"
            cxx_flags = f"-stdlib=libc++ -I{llvm_include}/c++/v1"
            linker_flags = (
                f"-stdlib=libc++ -L{llvm_lib} -L{prefix}/lib "
                f"-L{prefix}/opt/jpeg-turbo/lib"
            )

        flags.extend([
            f"-DCMAKE_CXX_FLAGS={cxx_flags}",
            f"-DCMAKE_EXE_LINKER_FLAGS={linker_flags}",
            f"-DCMAKE_SHARED_LINKER_FLAGS={linker_flags}",
        ])

        return flags

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

        normalized_cmd: str | list[str]
        if shell:
            normalized_cmd = cmd
        elif isinstance(cmd, list):
            normalized_cmd = cmd
        else:
            normalized_cmd = list(cmd)

        result = subprocess.run(
            normalized_cmd,
            shell=shell,
            cwd=cwd,
            env=full_env,
            text=True,
            capture_output=True,
            check=False,
        )

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
        if shutil.which("zypper"):
            return PackageManager.ZYPPER

        # If prefer_system_pkg but no system manager found, try Homebrew
        if self.brew_prefix is not None:
            return PackageManager.HOMEBREW

        return PackageManager.UNKNOWN

    def install_deps(self) -> None:
        """Install build dependencies using detected package manager."""
        pkg_manager = self.detect_package_manager()
        self._logger.info("Using package manager: %s", pkg_manager)

        if pkg_manager in (
            PackageManager.DNF,
            PackageManager.APT,
            PackageManager.PACMAN,
            PackageManager.ZYPPER,
        ):
            if _is_root():
                self._logger.info("Running as root, no sudo needed")
            else:
                self._logger.info(
                    "Running as user, will use sudo for package installation"
                )

        match pkg_manager:
            case PackageManager.HOMEBREW:
                self._install_homebrew_deps()
            case PackageManager.DNF:
                self._install_dnf_deps()
            case PackageManager.APT:
                self._install_apt_deps()
            case PackageManager.PACMAN:
                self._install_pacman_deps()
            case PackageManager.ZYPPER:
                self._install_zypper_deps()
            case PackageManager.UNKNOWN:
                raise BuildError(
                    "No supported package manager found. "
                    "Install build deps manually and use --skip-deps"
                )

    def _install_homebrew_deps(self) -> None:
        """Install dependencies via Homebrew."""
        self._logger.info("Homebrew prefix: %s", self.brew_prefix)

        packages = [
            # Build tools
            "cmake",
            "ninja",
            "pkg-config",
            "llvm",
            # Libraries
            "aom",
            "jpeg-turbo",
            "libpng",
            # Utilities
            "git",
        ]
        self.run(["brew", "install", *packages])

    def _install_dnf_deps(self) -> None:
        """Install dependencies via DNF (Fedora/RHEL)."""
        packages = [
            # Compiler toolchain
            "clang",
            "llvm",
            "libcxx",
            "libcxx-devel",
            "libcxxabi-devel",
            # Build tools
            "cmake",
            "ninja-build",
            "pkg-config",
            # Libraries
            "libaom-devel",
            "libjpeg-turbo-devel",
            "libpng-devel",
            # Utilities
            "git",
        ]
        self.run(_privileged_cmd(["dnf", "install", "-y", *packages]))

    def _install_apt_deps(self) -> None:
        """Install dependencies via APT (Debian/Ubuntu)."""
        packages = [
            # Compiler toolchain
            "clang",
            "llvm",
            "libc++-dev",
            "libc++abi-dev",
            # Build tools
            "cmake",
            "ninja-build",
            "pkg-config",
            # Libraries
            "libaom-dev",
            "libjpeg-turbo8-dev",
            "libpng-dev",
            # Utilities
            "git",
        ]
        self.run(_privileged_cmd(["apt-get", "update"]))
        self.run(_privileged_cmd(["apt-get", "install", "-y", *packages]))

    def _install_pacman_deps(self) -> None:
        """Install dependencies via Pacman (Arch)."""
        packages = [
            # Compiler toolchain
            "clang",
            "llvm",
            "libc++",
            # Build tools
            "cmake",
            "ninja",
            "pkgconf",
            # Libraries
            "aom",
            "libjpeg-turbo",
            "libpng",
            # Utilities
            "git",
        ]
        self.run(
            _privileged_cmd(["pacman", "-S", "--needed", "--noconfirm", *packages])
        )

    def _install_zypper_deps(self) -> None:
        """Install dependencies via Zypper (openSUSE)."""
        packages = [
            # Compiler toolchain
            "clang",
            "llvm",
            "libc++-devel",
            "libc++abi-devel",
            # Build tools
            "cmake",
            "ninja",
            "pkgconf-pkg-config",
            # Libraries
            "libaom-devel",
            "libjpeg8-devel",
            "libpng16-devel",
            # Utilities
            "git",
        ]
        self.run(_privileged_cmd(["zypper", "-n", "install", *packages]))

    def _clone_repo(self, name: str, url: str, tag: str) -> Path:
        """Clone or update a library repository."""
        dest = self.config.sources_dir / name

        if dest.exists():
            self._logger.info("Using existing %s source", name)
            with contextlib.chdir(dest):
                self.run(["git", "fetch", "--tags"], check=False)
                self.run(["git", "checkout", tag], check=False)
        else:
            self._logger.info("Cloning %s (tag: %s)...", name, tag)
            self.config.sources_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.chdir(self.config.sources_dir):
                self.run([
                    "git", "clone",
                    "--depth", "1",
                    "--branch", tag,
                    url,
                    name,
                ])

        return dest

    def build_libavif(self) -> None:
        """Build libavif with apps from source."""
        self._logger.info("Building libavif v1.3.0...")

        src = self._clone_repo(
            "libavif",
            "https://github.com/AOMediaCodec/libavif.git",
            "v1.3.0",
        )

        # Build RPATH for runtime library discovery
        rpath_dirs = [str(self.config.install_prefix / "lib")]
        if self.brew_prefix:
            rpath_dirs.extend([
                f"{self.brew_prefix}/lib",
                f"{self.brew_prefix}/opt/llvm/lib",
            ])
        rpath = ";".join(rpath_dirs)

        cmake_args = [
            "cmake", "-B", "build", "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={self.config.install_prefix}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            # Fix for CMake 4.x compatibility with older libyuv
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            # Set RPATH so binaries find Homebrew/system libs at runtime
            f"-DCMAKE_INSTALL_RPATH={rpath}",
            f"-DCMAKE_BUILD_RPATH={rpath}",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
            # The key flag - builds avifenc, avifdec, avifgainmaputil
            "-DAVIF_BUILD_APPS=ON",
            "-DAVIF_BUILD_TESTS=OFF",
            # Use system libraries for everything
            "-DAVIF_CODEC_AOM=SYSTEM",
            "-DAVIF_CODEC_SVT=OFF",
            "-DAVIF_CODEC_DAV1D=OFF",
            "-DAVIF_JPEG=SYSTEM",
            "-DAVIF_ZLIBPNG=SYSTEM",
            # libyuv for color conversion (let libavif fetch it)
            "-DAVIF_LIBYUV=LOCAL",
            *self._cmake_compiler_flags(),
        ]

        with contextlib.chdir(src):
            self.run(cmake_args)
            self.run(["cmake", "--build", "build", f"-j{self.config.jobs}"])
            self.run(["cmake", "--install", "build"])

    def verify(self) -> None:
        """Verify the build succeeded."""
        bin_dir = self.config.install_prefix / "bin"

        binaries = [
            ("avifenc", ["--version"]),
            ("avifdec", ["--version"]),
            ("avifgainmaputil", ["--help"]),
        ]

        failed = []

        for name, args in binaries:
            binary = bin_dir / name

            if not binary.exists():
                failed.append(f"{name}: not found at {binary}")
                continue

            if not os.access(binary, os.X_OK):
                failed.append(f"{name}: not executable")
                continue

            # Run version/help check
            try:
                result = self.run([str(binary), *args], check=False)
                # avifgainmaputil --help returns non-zero, that's OK
                if result.returncode != 0 and name != "avifgainmaputil":
                    failed.append(f"{name}: returned non-zero exit code")
            except BuildError as exc:
                failed.append(f"{name}: {exc}")

        if failed:
            for msg in failed:
                self._logger.error(msg)
            raise BuildError("Build verification failed")

        # Success message
        c = _AnsiColor
        print(f"""
{c.GREEN}{"=" * 50}{c.RESET}
{c.GREEN}{c.BOLD}AVIF Tools Build Successful!{c.RESET}
{c.GREEN}{"=" * 50}{c.RESET}

{c.BOLD}Binaries installed to:{c.RESET} {bin_dir}
  - avifenc          (AVIF encoder)
  - avifdec          (AVIF decoder)
  - avifgainmaputil  (HDR gain map utility)

{c.BOLD}Usage examples:{c.RESET}
  {bin_dir}/avifenc input.png output.avif
  {bin_dir}/avifdec input.avif output.png
  {bin_dir}/avifgainmaputil --help

{c.GRAY}All build artifacts are in: {self.config.build_dir}{c.RESET}
""")

    def build_all(self) -> None:
        """Execute full build pipeline."""
        # Log compiler choice
        build_env = self._build_env()
        cc = build_env.get("CC", "cc")
        cxx = build_env.get("CXX", "c++")
        self._logger.info("Using compilers: CC=%s, CXX=%s", cc, cxx)

        if not self.config.skip_deps:
            self.install_deps()

        self.build_libavif()
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
        description="Build AVIF tools (avifenc, avifdec, avifgainmaputil) from source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Builds libavif with AVIF_BUILD_APPS=ON to produce command-line tools.
All dependencies (aom, zlib, libpng, libjpeg-turbo) come from system packages.

Environment variables:
  BUILD_DIR        Build directory (default: ./avif_build)
  INSTALL_PREFIX   Installation prefix (default: ./avif_build)
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
        help="Build directory (default: ./avif_build)",
    )
    parser.add_argument(
        "--install-prefix",
        type=Path,
        default=None,
        metavar="DIR",
        help="Installation prefix (default: ./avif_build)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
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
        "-v",
        "--verbose",
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
{c.BOLD}{"=" * 50}
AVIF Tools Builder v{__version__}
{"=" * 50}{c.RESET}

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
