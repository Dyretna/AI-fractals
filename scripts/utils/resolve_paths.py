from pathlib import Path


def resolve_output_paths(cfg: dict, project_root: Path):
    job_type = cfg.get("job_type", None)

    if job_type == "region":
        # region builder has ONE output-dir
        if "output_dir" not in cfg:
            raise ValueError("Config missing 'output_dir'.")
        cfg["output_path"] = (project_root / cfg["output_dir"]).resolve()
        return cfg

    elif job_type == "shoreline":
        # shoreline has FIVE paths
        required = [
            "region_raw_dir",
            "region_evaluated_dir",
            "region_rejected_dir",
            "shoreline_evaluated_dir",
            "shoreline_rejected_dir",
        ]
        for key in required:
            if key not in cfg:
                raise ValueError(f"Config missing '{key}'.")

        cfg["region_raw_dir"] = Path(project_root / cfg["region_raw_dir"]).resolve()
        cfg["region_evaluated_dir"] = Path(
            project_root / cfg["region_evaluated_dir"]
        ).resolve()
        cfg["region_rejected_dir"] = Path(
            project_root / cfg["region_rejected_dir"]
        ).resolve()
        cfg["shoreline_evaluated_dir"] = Path(
            project_root / cfg["shoreline_evaluated_dir"]
        ).resolve()
        cfg["shoreline_rejected_dir"] = Path(
            project_root / cfg["shoreline_rejected_dir"]
        ).resolve()
        return cfg

    elif job_type == "rgb":
        if "output_dir" not in cfg:
            raise ValueError("Config missing 'output_dir'.")

        cfg["region_evaluated_dir"] = Path(
            project_root / cfg["region_evaluated_dir"]
        ).resolve()
        cfg["output_dir"] = (project_root / cfg["output_dir"]).resolve()
        return cfg
    else:
        raise ValueError(f"Unknown job_type: {job_type}")
