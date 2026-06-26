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

            # 2. Setup stable python wrappers to make PyCharm work without plugins
            echo "Creating Python interpreter wrappers for IDE compatibility..."
            rm -f .venv/bin/python .venv/bin/python3 .venv/bin/python-real .venv/bin/python3-real
            ln -sfn ${pkgs.python3}/bin/python3 .venv/bin/python-real
            ln -sfn ${pkgs.python3}/bin/python3 .venv/bin/python3-real

            # Determine python version dynamically (e.g. python3.13)
            PYTHON_VERSION=$(${pkgs.python3}/bin/python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")

            # Helper function to generate interpreter wrapper scripts
            write_wrapper() {
              local target="$1"
              local real_bin="$2"
              echo "#!/bin/sh" > "$target"
              echo "export VIRTUAL_ENV=\"\$(cd \"\$(dirname \"\$0\")/..\" && pwd)\"" >> "$target"
              echo "export PYTHONPATH=\"\$VIRTUAL_ENV/lib/$PYTHON_VERSION/site-packages:\$PYTHONPATH\"" >> "$target"
              echo "export LD_LIBRARY_PATH=\"${libPath}:\$LD_LIBRARY_PATH\"" >> "$target"
              echo "exec \"\$VIRTUAL_ENV/bin/$real_bin\" \"\$@\"" >> "$target"
              chmod +x "$target"
            }

            write_wrapper .venv/bin/python python-real
            write_wrapper .venv/bin/python3 python3-real

            echo ""
            echo "You can run your project using uv:"
            echo "  uv run python src/detection/yolo.py"
            echo "========================================================="
          '';
        };
      });
    };
}
