{
  description = "llama.cpp router with web dashboard and scheduler";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: rec {
        llama-router = pkgs.callPackage ./package.nix { };
        default = llama-router;
      });

      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            fastapi
            uvicorn
            httpx
            aiosqlite
            pynvml
            # uvicorn[standard] extras — the app runs uvicorn with defaults.
            uvloop
            httptools
            websockets
            watchfiles
            python-dotenv
            pyyaml
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            shellHook = ''
              echo "llama-router dev shell — python $(${pythonEnv}/bin/python3 --version | cut -d' ' -f2)"
              echo "  app:  python src/main.py"
              echo "  demo: python frontend-demo/launch.py   # -> http://127.0.0.1:11500/dash"
            '';
          };
        });

      # The module resolves its packages via the consumer's `pkgs`,
      # so it follows whatever nixpkgs the importing flake uses.
      nixosModules = rec {
        llama-router = import ./module.nix;
        default = llama-router;
      };
    };
}
