import os, shutil

def create_out_dir(output_dir: str, backup_path: list[str]):
    backup_dir = os.path.join(output_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # 백업 설정
    print(f"[INFO] Backup setting")
    for path in backup_path:
        target_path = os.path.join(backup_dir, os.path.basename(path))
        shutil.copy(path, target_path)
        print("    -->", f"move to {target_path}")

