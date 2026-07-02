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

      # The module resolves its packages via the consumer's `pkgs`,
      # so it follows whatever nixpkgs the importing flake uses.
      nixosModules = rec {
        llama-router = import ./module.nix;
        default = llama-router;
      };
    };
}
