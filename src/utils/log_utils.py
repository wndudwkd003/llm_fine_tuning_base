import os, matplotlib.pyplot as plt

def save_training_curves(trainer, output_dir: str,
                         loss_fname: str = "loss_curve.png",
                         acc_fname: str = "accuracy_curve.png"):
    """
    Trainer 로그를 이용해 학습·평가 손실·정확도 그래프 두 장을 저장합니다.

    Parameters
    ----------
    trainer : transformers.Trainer 또는 trl.SFTTrainer
        학습을 마친 Trainer 객체
    output_dir : str
        그래프를 저장할 폴더 경로
    loss_fname : str, optional
        손실 그래프 파일 이름, 기본값 'loss_curve.png'
    acc_fname : str, optional
        정확도 그래프 파일 이름, 기본값 'accuracy_curve.png'
    """
    log_history = trainer.state.log_history

    train_steps, train_losses, train_accs = [], [], []
    eval_steps, eval_losses, eval_accs = [], [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
            if "mean_token_accuracy" in entry:
                train_accs.append(entry["mean_token_accuracy"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])
            if "mean_token_accuracy" in entry:
                eval_accs.append(entry["mean_token_accuracy"])

    os.makedirs(output_dir, exist_ok=True)

    # 손실 그래프
    plt.figure()
    plt.plot(train_steps, train_losses, label="train_loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval_loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, loss_fname), dpi=300)
    plt.close()

    # 정확도 그래프
    if train_accs or eval_accs:
        plt.figure()
        if train_accs:
            plt.plot(train_steps[:len(train_accs)], train_accs, label="train_accuracy")
        if eval_accs:
            plt.plot(eval_steps, eval_accs, label="eval_accuracy")
        plt.xlabel("step"); plt.ylabel("token_accuracy"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, acc_fname), dpi=300)
        plt.close()
