{
  description = "Nix flake for Acadia-AI-E26 Python development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Supported systems for development
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      # Helper function to generate attributes for each system
      forEachSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f rec {
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # Allow unfree packages if needed
        };
        
        # Library path for PyPI wheels on NixOS (includes X11, GL, and DBus)
        libPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          pkgs.zlib
          pkgs.glib
          pkgs.libGL
          pkgs.libx11
          pkgs.libxext
          pkgs.libxrender
          pkgs.libxi
          pkgs.libxcb
          pkgs.libxkbcommon
          pkgs.dbus.lib
        ];
      });
    in
    {
      # Development shell providing base Python, uv, and system libraries
      devShells = forEachSystem ({ pkgs, libPath, ... }: {
        default = pkgs.mkShell {
          name = "acadia-ai-e26-env";

          # Packages available in the shell
          packages = [
            pkgs.python3
            pkgs.uv
          ];

          # Shell initialization hook
          shellHook = ''
            # Setup LD_LIBRARY_PATH for PyPI wheels inside the shell
            export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"

            echo "========================================================="
            echo "   Welcome to the Acadia AI E26 Development Environment   "
            echo "========================================================="
            echo "Using base Python and uv."
            echo ""

            # 1. Automatically create/sync the virtual environment using uv
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment and syncing dependencies..."
              uv venv
              VIRTUAL_ENV=.venv uv sync
            else
              echo "Syncing dependencies with uv..."
              VIRTUAL_ENV=.venv uv sync
            fi

            echo ""
            echo "You can run your project using uv:"
            echo "  uv run python src/detection/yolo.py"
            echo "========================================================="
          '';
        };
      });
    };
}
