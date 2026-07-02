{ config, lib, pkgs, ... }:

let
  cfg = config.services.llama-router;

  iniAtom = with lib.types; oneOf [ bool int float str ];

  # Router JSON: pick num_instance per model out of the shared `models` attrset.
  routerConfig = pkgs.writeText "llama-router-config.json" (builtins.toJSON {
    LLM = lib.mapAttrs (_: m: { num_instance = m.num_instance or 1; }) cfg.models;
    ROUTER = cfg.routerSettings;
    "API-port" = cfg.port;
    "LLM-base-port" = cfg.llmBasePort;
    "llama-server-executable" = "${cfg.llamaCpp}/bin/llama-server";
  });

  # Preset INI: drop num_instance (router-only) from each model, prepend the "[*]" globals
  presetsFormat = pkgs.formats.ini {
    mkKeyValue = lib.generators.mkKeyValueDefault {} " = ";
  };
  presetsIni = presetsFormat.generate "llama-presets.ini" (
    lib.optionalAttrs (cfg.presetGlobals != {}) { "*" = cfg.presetGlobals; }
    // lib.mapAttrs (_: m: removeAttrs m [ "num_instance" ]) cfg.models
  );

  emb = cfg.embedding;
in
{
  options.services.llama-router = {
    enable = lib.mkEnableOption "llama-router, a llama.cpp router with web dashboard and scheduler";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix {};
      defaultText = lib.literalExpression "pkgs.callPackage ./package.nix {}";
      description = "The llama-router package to run.";
    };

    llamaCpp = lib.mkOption {
      type = lib.types.package;
      default = pkgs.llama-cpp;
      defaultText = lib.literalExpression "pkgs.llama-cpp";
      description = "llama.cpp package providing the llama-server binary spawned by the router.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address the router API binds to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 11434;
      description = "Port for the router API.";
    };

    llmBasePort = lib.mkOption {
      type = lib.types.port;
      default = 30000;
      description = "First port used for spawned llama-server instances.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "llama-router";
      description = "User to run the services as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "llama-router";
      description = "Group to run the services as.";
    };

    extraGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "video" "render" ];
      description = "Extra groups for the service user (GPU access).";
    };

    modelDirs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      example = [ "/data/llm-models" ];
      description = "Directories created at boot (0755, service-owned) for model storage.";
    };

    presetGlobals = lib.mkOption {
      type = lib.types.attrsOf iniAtom;
      default = {};
      example = { jinja = true; fa = true; ngl = 99; };
      description = "llama.cpp settings applied to every preset (the \"[*]\" wildcard section).";
    };

    models = lib.mkOption {
      type = lib.types.attrsOf (lib.types.attrsOf iniAtom);
      default = {};
      example = lib.literalExpression ''
        {
          "Qwen3-4B" = {
            num_instance = 1;
            model = "/data/llm-models/Qwen3-4B-Q8_0.gguf";
            c = 65536;
            parallel = 4;
          };
        }
      '';
      description = ''
        Model presets. Each attribute becomes a llama.cpp presets.ini section;
        `num_instance` is consumed by the router and stripped from the INI.
      '';
    };

    routerSettings = lib.mkOption {
      type = lib.types.attrsOf iniAtom;
      default = {
        HEALTH_CHECK_INTERVAL = 1.0;
        HEALTH_CHECK_TIMEOUT = 30.0;
        UNLOAD_POLL_INTERVAL = 0.5;
        UNLOAD_POLL_TIMEOUT = 60.0;
        LOAD_POLL_INTERVAL = 1.0;
        LOAD_POLL_TIMEOUT = 120.0;
        START_RETRIES = 3;
        GRACEFUL_KILL_TIMEOUT = 5.0;
      };
      description = "ROUTER section of the router config (scheduler timings).";
    };

    embedding = {
      enable = lib.mkEnableOption "a standalone llama.cpp embedding server alongside the router";

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        description = "Address the embedding server binds to.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 11435;
        description = "Port for the embedding server.";
      };

      model = lib.mkOption {
        type = lib.types.str;
        example = "/data/llm-models/nomic-embed-text-v2-moe.Q4_0.gguf";
        description = "Path to the embedding model GGUF.";
      };

      pooling = lib.mkOption {
        type = lib.types.str;
        default = "cls";
        description = "Pooling mode passed to llama-server.";
      };

      ctxSize = lib.mkOption {
        type = lib.types.int;
        default = 2048;
        description = "Context size for the embedding server.";
      };

      parallel = lib.mkOption {
        type = lib.types.int;
        default = 4;
        description = "Number of parallel slots.";
      };

      nGpuLayers = lib.mkOption {
        type = lib.types.int;
        default = -1;
        description = "GPU layers (-1 = all).";
      };

      extraFlags = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [];
        description = "Extra flags appended to the llama-server command line.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      inherit (cfg) extraGroups;
    };
    users.groups.${cfg.group} = {};

    systemd.tmpfiles.rules = map (d: "d ${d} 0755 ${cfg.user} ${cfg.group} -") cfg.modelDirs;

    systemd.services.llama-router = {
      description = "LLaMA.cpp Router";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      environment = {
        ROUTER_CONFIG_PATH = "${routerConfig}";
        LLAMA_PRESETS_PATH = "${presetsIni}";
        ROUTER_HOST = cfg.host;
        HISTORY_DB_PATH = "/var/lib/llama-router/monitor/history.db";
      };
      serviceConfig = {
        ExecStart = lib.getExe cfg.package;
        WorkingDirectory = "/var/lib/llama-router";
        StateDirectory = "llama-router";
        User = cfg.user;
        Group = cfg.group;
        Restart = "on-failure";
        RestartSec = "5s";
      };
    };

    systemd.services.llama-embedding = lib.mkIf emb.enable {
      description = "LLaMA.cpp Embedding Server";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = lib.concatStringsSep " " ([
          "${cfg.llamaCpp}/bin/llama-server"
          "--host ${emb.host} --port ${toString emb.port}"
          "--model ${emb.model}"
          "--embedding --pooling ${emb.pooling}"
          "--ctx-size ${toString emb.ctxSize} --parallel ${toString emb.parallel} --n-gpu-layers ${toString emb.nGpuLayers}"
        ] ++ emb.extraFlags);
        User = cfg.user;
        Group = cfg.group;
        Restart = "on-failure";
        RestartSec = "5s";
      };
    };
  };
}
