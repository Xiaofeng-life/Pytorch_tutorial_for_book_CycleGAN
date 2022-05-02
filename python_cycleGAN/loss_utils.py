import os


class LossWriter():
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def add(self, loss_name, loss, i):
        with open(os.path.join(self.save_dir, loss_name + ".txt"), mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


if __name__ == "__main__":
    pass
