import trainer as T

def main():
    trainer = T.Trainer()
    for _ in range(5):
        trainer.collect_training_data(True, std = 0.5)
    for _ in range(1000):
        trainer.single_train_step()

if __name__ == '__main__':
    main()
