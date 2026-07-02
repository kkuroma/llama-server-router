{ lib, stdenvNoCC, python3, makeWrapper }:

let
  pythonEnv = python3.withPackages (ps: with ps; [
    fastapi
    uvicorn
    httpx
    aiosqlite
    pynvml
  ]);
in
stdenvNoCC.mkDerivation {
  pname = "llama-router";
  version = "0.1.0";

  src = ./src;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/share/llama-router
    cp -r . $out/share/llama-router
    makeWrapper ${pythonEnv}/bin/python3 $out/bin/llama-router \
      --add-flags $out/share/llama-router/main.py
    runHook postInstall
  '';

  meta = {
    description = "llama.cpp router with web dashboard and scheduler";
    homepage = "https://git.kuroma.dev/kkuroma/llama-router";
    license = lib.licenses.gpl3Only;
    mainProgram = "llama-router";
    platforms = lib.platforms.linux;
  };
}
