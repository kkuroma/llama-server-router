{ config, lib, pkgs, ... }:

let
  cfg = config.services.llama-router;

  iniAtom = with lib.types; oneOf [ bool int float str ];

  # Router JSON: pick num_instance + gpus + reasoning_effort per model out of the shared `models` attrset.
  routerConfig = pkgs.writeText "llama-router-config.json" (builtins.toJSON {
    LLM = lib.mapAttrs (_: m:
      { num_instance = m.num_instance or 1; }
      // lib.optionalAttrs (m ? gpus) { inherit (m) gpus; }
      // lib.optionalAttrs (m ? reasoning_effort) { inherit (m) reasoning_effort; }
      // lib.optionalAttrs (m ? reasoning_effort_default) { inherit (m) reasoning_effort_default; }
    ) cfg.models;
    ROUTER = {
      MAX_MODELS_PER_GPU = cfg.maxModelsPerGpu;
      EVICTION_POLICY = cfg.evictionPolicy;
      QUEUE_FORCE_LOAD_TIMEOUT = cfg.queueForceLoadTimeout;
      QUEUE_HEAD_GRACE = cfg.queueHeadGrace;
    } // lib.optionalAttrs (cfg.gpuCount != null) { NUM_GPUS = cfg.gpuCount; }
      // cfg.routerSettings;
    "API-port" = cfg.port;
    "LLM-base-port" = cfg.llmBasePort;
    "llama-server-executable" = "${cfg.llamaCpp}/bin/llama-server";
  });

  # Preset INI: drop num_instance + gpus + reasoning_effort + reasoning_effort_default
  # (router-only) from each model, prepend the "[*]" globals. Physical placement is NOT
  # emitted as a `device` key: the router masks each llama-server to its `gpus` via
  # CUDA_VISIBLE_DEVICES, which both pins compute and keeps ggml's per-device
  # context/buffers off other GPUs. `gpus` is therefore the single source of truth
  # for placement.
  #
  # A per-model cache cap lands here rather than in "[*]", and an explicit `cram`
  # on the model still wins because it merges last.
  mkPreset = name: m:
    lib.optionalAttrs (cfg.promptCache.ramMiBPerModel ? ${name})
      { cram = cfg.promptCache.ramMiBPerModel.${name}; }
    // removeAttrs m [ "num_instance" "gpus" "reasoning_effort" "reasoning_effort_default" ];
  presetsFormat = pkgs.formats.ini {
    mkKeyValue = lib.generators.mkKeyValueDefault {} " = ";
  };

  # Prompt cache keys, written into "[*]" so presetGlobals and per-model keys still win.
  pc = cfg.promptCache;
  cacheGlobals =
    { cache-prompt = pc.enable; cram = pc.ramMiB; }
    // lib.optionalAttrs (pc.reuseChunk != 0) { cache-reuse = pc.reuseChunk; }
    // lib.optionalAttrs (pc.checkpoints != null) { ctx-checkpoints = pc.checkpoints; }
    // lib.optionalAttrs (pc.checkpointMinStep != null) { checkpoint-min-step = pc.checkpointMinStep; };

  globals = cacheGlobals // cfg.presetGlobals;
  presetsIni = presetsFormat.generate "llama-presets.ini" (
    lib.optionalAttrs (globals != {}) { "*" = globals; }
    // lib.mapAttrs mkPreset cfg.models
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
      description = ''
        llama.cpp settings applied to every preset (the "[*]" wildcard section).
        Keys here override the ones `promptCache` generates, and a key on a model
        overrides both.
      '';
    };

    promptCache = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Whether llama.cpp reuses an already-processed prompt prefix instead of
          reprocessing it (`--cache-prompt`). Off means every request pays full
          prompt processing, so leave it on unless you are measuring against it.
        '';
      };

      ramMiB = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 8192;
        example = 16384;
        description = ''
          Host memory cap for conversations that are not in the slot right now
          (`--cache-ram`), per llama-server process. 0 disables the host cache, so
          only the conversation currently in a slot stays warm.

          Size it as (concurrent conversations) x (cost of one conversation at the
          prompt length you actually run). That cost is a fixed part plus a part that
          grows with the prompt, so a bytes-per-token figure taken from a short prompt
          understates a long one. Measured on a 3090 at `q4_0`: Gemma-4-26B-A4B is
          176 MiB plus 5.7 KiB/token, reaching 909 MiB at its full 131k window, and
          Wordslop-Qwen3.6-27B is 766 MiB plus 30.2 KiB/token, reaching 8500 MiB at
          its full 262k window. Over the cap, llama.cpp evicts the least recently
          used conversation, and returning to it costs a full reprocess. The cap is
          not an allocation: memory is used only as conversations arrive.
        '';
      };

      ramMiBPerModel = lib.mkOption {
        type = lib.types.attrsOf lib.types.ints.unsigned;
        default = {};
        example = { "Wordslop-Qwen3.6-27B" = 32768; };
        description = ''
          Per-model override of `ramMiB`, keyed by model name. Use it when one
          model's cache costs far more per conversation than the rest, either
          because its context window is longer or because more of its layers hold
          a KV cache that grows with the prompt. The cap is per llama-server
          process, so with `maxModelsPerGpu = 1` only the loaded model's cap is
          ever live and a large value here does not add to the others.

          Written into the model's own preset section, so it beats `ramMiB` and
          `presetGlobals.cram`; a literal `cram` on the model beats it in turn.
        '';
      };

      reuseChunk = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 0;
        description = ''
          Smallest chunk llama.cpp will try to salvage by KV shifting when the prompt
          changed before its tail (`--cache-reuse`). Ignored by models whose context
          cannot shift, which includes every sliding-window model; those log
          "cache_reuse is not supported by this context" at load and reprocess from
          the first changed token.
        '';
      };

      checkpoints = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.unsigned;
        default = null;
        description = "Context checkpoints kept per slot (`--ctx-checkpoints`), or null for the llama.cpp default.";
      };

      checkpointMinStep = lib.mkOption {
        type = lib.types.nullOr lib.types.ints.unsigned;
        default = null;
        description = "Minimum token spacing between context checkpoints (`--checkpoint-min-step`), or null for the llama.cpp default.";
      };
    };

    models = lib.mkOption {
      type = lib.types.attrsOf (lib.types.attrsOf (lib.types.either iniAtom (lib.types.listOf (lib.types.either lib.types.int lib.types.str))));
      default = {};
      example = lib.literalExpression ''
        {
          "Qwen3-4B" = {
            num_instance = 1;
            gpus = [ 0 1 ];
            reasoning_effort = [ "low" "medium" "xhigh" ];
            model = "/data/llm-models/Qwen3-4B-Q8_0.gguf";
            c = 65536;
            parallel = 4;
          };
        }
      '';
      description = ''
        Model presets. Each attribute becomes a llama.cpp presets.ini section;
        `num_instance`, `gpus`, `reasoning_effort`, and `reasoning_effort_default`
        are consumed by the router and stripped from the INI. `gpus` pins the model
        to GPU ids (omitted = GPU 0 only; "all" or -1 = every GPU) and is the single
        source of truth for physical placement: the router masks each llama-server
        to those GPUs via CUDA_VISIBLE_DEVICES, so no `device` key is emitted or
        needed. Set `reasoning_effort` to a list of supported effort levels (e.g.
        [ "low" "medium" "xhigh" ] for Qwen 3.8) to advertise the model's reasoning
        capabilities via the /v1/models endpoint. Optionally set
        `reasoning_effort_default` to override which level is advertised as the
        default (defaults to the first level in the list).
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

    queueHeadGrace = lib.mkOption {
      type = lib.types.numbers.nonnegative;
      default = 0;
      description = ''
        Seconds a head-of-queue request whose model is not loaded tolerates newer requests for
        a resident model being served ahead of it. At 0 arrival order is absolute and two models
        in alternation pay a swap each time; raising it trades that ordering for cache hits, and
        queueForceLoadTimeout remains the last resort behind it.
      '';
    };

    gpuCount = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      description = "Number of GPUs. null = autodetect via NVML, falling back to highest pinned id + 1.";
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
    assertions = [{
      assertion = lib.all (n: cfg.models ? ${n}) (lib.attrNames cfg.promptCache.ramMiBPerModel);
      message =
        "services.llama-router.promptCache.ramMiBPerModel names unknown model(s): "
        + lib.concatStringsSep ", "
            (lib.subtractLists (lib.attrNames cfg.models)
                               (lib.attrNames cfg.promptCache.ramMiBPerModel));
    }];

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
