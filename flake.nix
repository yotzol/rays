{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;  # CUDA
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            cudaPackages.cudatoolkit  # CUDA
            cudaPackages.cuda_cudart
            gcc13                     # C++ compiler
            cmake                     # build system
            ninja                     # fast builds
            libpng                    # PNG output
            imgui                     # user interface
            glfw                      # window and opengl context management
          ];
          # use gcc13 instead of system's gcc
          shellHook = ''
            export PATH=${pkgs.gcc13}/bin:$PATH
          '';
        };
      }
    );
}
