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

      # `nix develop` / direnv shell: the runtime python deps plus the lint and
      # type tools, so `python src/main.py`, ruff, black and pyright all resolve
      # on PATH the moment you cd in. Deps mirror package.nix's pythonEnv.
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            fastapi
            uvicorn
            httpx
            aiosqlite
            pynvml
          ]);
        in
        {
          default = pkgs.mkShell {
            DEV_SHELL = "llama-router";
            packages = [
              pythonEnv
              pkgs.ruff
              pkgs.black
              pkgs.pyright
            ];
            shellHook = ''
              export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
              [[ $- == *i* ]] && exec zsh
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
