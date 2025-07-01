import os, shutil

def create_out_dir(
    output_dir: str,
    train_type: str,
    model_id: str,
    additional_info: str = "",
    backup_path: list[str] = None
):
    target_dir = os.path.join(
        output_dir,
        train_type,
        model_id,
        additional_info
    )

    backup_dir = os.path.join(target_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # 백업 설정
    print(f"[INFO] Backup setting")
    for path in backup_path:
        target_path = os.path.join(backup_dir, os.path.basename(path))
        shutil.copy(path, target_path)
        print("    -->", f"move to {target_path}")

    return target_dir




# def output_path_record(output_dir: str):
#     with open(os.path.join(output_dir, "output_path.txt"), "w") as f:
#         f.write(output_dir)

