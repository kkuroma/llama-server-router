{ config, lib, pkgs, ... }:

let
  cfg = config.services.llama-router;

  iniAtom = with lib.types; oneOf [ bool int float str ];

  # gpus = "all" / -1 / a list containing -1 means "every GPU"
  gpusIsAll = g: g == "all" || g == -1 || (builtins.isList g && builtins.elem (-1) g);

  # Router JSON: pick num_instance + gpus per model out of the shared `models` attrset.
  routerConfig = pkgs.writeText "llama-router-config.json" (builtins.toJSON {
    LLM = lib.mapAttrs (_: m:
      { num_instance = m.num_instance or 1; }
      // lib.optionalAttrs (m ? gpus) { inherit (m) gpus; }
    ) cfg.models;
    ROUTER = {
      MAX_MODELS_PER_GPU = cfg.maxModelsPerGpu;
      EVICTION_POLICY = cfg.evictionPolicy;
      QUEUE_FORCE_LOAD_TIMEOUT = cfg.queueForceLoadTimeout;
    } // lib.optionalAttrs (cfg.gpuCount != null) { NUM_GPUS = cfg.gpuCount; }
      // cfg.routerSettings;
    "API-port" = cfg.port;
    "LLM-base-port" = cfg.llmBasePort;
    "llama-server-executable" = "${cfg.llamaCpp}/bin/llama-server";
  });

  # Preset INI: drop num_instance + gpus (router-only) from each model, prepend
  # the "[*]" globals. An explicit gpus list pins the model physically via the
  # llama-server `device` key (unless the model already sets device itself);
  # "all"/-1 emits no device key = llama-server's default (all GPUs).
  mkPreset = m:
    removeAttrs m [ "num_instance" "gpus" ]
    // lib.optionalAttrs (m ? gpus && builtins.isList m.gpus && !gpusIsAll m.gpus && !(m ? device)) {
      device = lib.concatMapStringsSep "," (i: "${cfg.gpuDevicePrefix}${toString i}") m.gpus;
    };
  presetsFormat = pkgs.formats.ini {
    mkKeyValue = lib.generators.mkKeyValueDefault {} " = ";
  };
  presetsIni = presetsFormat.generate "llama-presets.ini" (
    lib.optionalAttrs (cfg.presetGlobals != {}) { "*" = cfg.presetGlobals; }
    // lib.mapAttrs (_: mkPreset) cfg.models
  );
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
      type = lib.types.attrsOf (lib.types.attrsOf (lib.types.either iniAtom (lib.types.listOf lib.types.int)));
      default = {};
      example = lib.literalExpression ''
        {
          "Qwen3-4B" = {
            num_instance = 1;
            gpus = [ 0 1 ];
            model = "/data/llm-models/Qwen3-4B-Q8_0.gguf";
            c = 65536;
            parallel = 4;
          };
        }
      '';
      description = ''
        Model presets. Each attribute becomes a llama.cpp presets.ini section;
        `num_instance` and `gpus` are consumed by the router and stripped from
        the INI. `gpus` pins the model to GPU ids (omitted = GPU 0 only;
        "all" or -1 = every GPU) — an explicit list also emits a llama-server
        `device` key so physical placement matches the scheduler's accounting.
      '';
    };

    maxModelsPerGpu = lib.mkOption {
      type = lib.types.ints.positive;
      default = 1;
      description = "How many models may stay resident PER GPU before the router evicts.";
    };

    evictionPolicy = lib.mkOption {
      type = lib.types.enum [ "lru" "fifo" ];
      default = "lru";
      description = "Which resident model to evict on overflow: least-recently-requested (lru) or earliest-loaded (fifo).";
    };

    queueForceLoadTimeout = lib.mkOption {
      type = lib.types.numbers.positive;
      default = 300;
      description = "Seconds the head-of-queue request may wait for an unloaded model before the router force-loads it past newer cache-hit requests.";
    };

    gpuCount = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      description = "Number of GPUs. null = autodetect via NVML, falling back to highest pinned id + 1.";
    };

    gpuDevicePrefix = lib.mkOption {
      type = lib.types.str;
      default = "CUDA";
      description = "Backend prefix used when deriving llama-server device names from gpu ids (CUDA0, ROCm0, ...).";
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
  };
}
