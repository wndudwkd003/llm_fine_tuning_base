import os, shutil
from src.utils.print_utils import printi

def create_out_dir(
    output_dir: str,
    train_type: str,
    model_id: str,
    additional_info: str = "",
    backup_path: list[str] = None,
    current_stage: str = ""
):

    model_id = slash_remove(model_id)


    current_stage = f"_{current_stage}" if current_stage != "" else ""

    target_name = f"{model_id}_{train_type}{f'_{additional_info}{current_stage}' if additional_info else ''}"
    print(f"{target_name}")

    target_dir = os.path.join(
        output_dir,
        target_name
    )

    backup_dir = os.path.join(target_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # 백업 설정
    printi(f"Backup setting")
    for path in backup_path:
        target_path = os.path.join(backup_dir, os.path.basename(path))
        shutil.copy(path, target_path)
        print("    -->", f"move to {target_path}")

    return target_dir, target_name



def slash_remove(path: str):
    return path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


# def output_path_record(output_dir: str):
#     with open(os.path.join(output_dir, "output_path.txt"), "w") as f:
#         f.write(output_dir)

