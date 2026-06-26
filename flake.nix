{
  description = "Nix flake for Acadia-AI-E26 Python development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    acadia-src = {
      url = "github:ElliotScher/Acadia-AI-E26";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, pyproject-nix, acadia-src }:
    let
      # Supported systems
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      # Helper function to generate attributes for each system
      forEachSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f rec {
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # Allow unfree packages like CUDA/PyTorch
        };
        
        # Load the pyproject.toml project metadata dynamically from the fetched source input
        project = pyproject-nix.lib.project.loadPyproject {
          projectRoot = acadia-src;
        };
        
        # Define a Python environment rendering the dependencies directly from pyproject.toml
        pythonEnv = pkgs.python3.withPackages (ps: 
          project.renderers.withPackages { 
            python = pkgs.python3; 
          } ps
        );
      });
    in
    {
      # Development shell to work on the project using UV/Python
      devShells = forEachSystem ({ pkgs, pythonEnv, ... }: {
        default = pkgs.mkShell {
          name = "acadia-ai-e26-env";

          # Packages available in the shell
          packages = [
            pythonEnv
            pkgs.uv
            pkgs.git # Ensure git is available to clone the submodule
          ];

          # Shell initialization hook
          shellHook = ''
            echo "========================================================="
            echo "   Welcome to the Acadia AI E26 Development Environment   "
            echo "========================================================="
            echo "Python is automatically loaded with all packages from pyproject.toml"
            echo "fetched directly from the GitHub input source."
            echo ""
            
            # Automatically clone/initialize the submodule if it's not present locally
            if [ ! -f "app/pyproject.toml" ]; then
              echo "Submodule 'app' not detected. Initializing submodules..."
              git submodule update --init --recursive
            else
              echo "Submodule 'app' is present."
            fi
            
            echo ""
            echo "You can run your project using uv:"
            echo "  uv run python src/spypoint-scraper/scraper.py"
            echo "  uv run python src/detection/yolo.py"
            echo "========================================================="
            
            # Setup LD_LIBRARY_PATH to avoid libGL / glib loading errors 
            # if you run PyPI wheels (like opencv-python or torch) inside the virtualenv.
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              pkgs.glib
              pkgs.libGL
              pkgs.libx11
              pkgs.libxext
              pkgs.libxrender
              pkgs.libxi
            ]}:$LD_LIBRARY_PATH"
          '';
        };
      });
    };
}
