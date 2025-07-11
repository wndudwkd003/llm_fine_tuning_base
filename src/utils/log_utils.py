import os, matplotlib.pyplot as plt

ACC_KEYS = ["eval_accuracy", "eval_token_accuracy", "mean_token_accuracy"]

def save_training_curves(trainer, output_dir: str,
                         loss_fname: str = "loss_curve.png",
                         acc_fname: str = "accuracy_curve.png"):
    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_accs = [], [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue

        # 훈련 손실
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])

        # 평가 손실·정확도
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
            # NaN 값 필터링
            if not (eval_loss != eval_loss):  # NaN 체크 (NaN != NaN은 True)
                eval_steps.append(step)
                eval_losses.append(eval_loss)

            # accuracy는 별도로 처리
            for k in ACC_KEYS:
                if k in entry:
                    eval_accs.append(entry[k])
                    break

    os.makedirs(output_dir, exist_ok=True)

    # ── Loss 곡선 ──
    plt.figure()
    plt.plot(train_steps, train_losses, label="train_loss")
    if eval_losses:  # NaN이 필터링된 후 데이터가 있다면
        plt.plot(eval_steps, eval_losses, label="eval_loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, loss_fname), dpi=300)
    plt.close()

    # ── Accuracy 곡선 ──
    if eval_accs:
        plt.figure()
        # accuracy용 step은 모든 eval step 사용
        eval_acc_steps = [entry.get("step") for entry in log_history if any(k in entry for k in ACC_KEYS)]
        plt.plot(eval_acc_steps, eval_accs, label="eval_accuracy")
        plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, acc_fname), dpi=300)
        plt.close()
