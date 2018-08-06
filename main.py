from main_train import *
import nsml

if __name__ == '__main__':
    nsml.report(
        summary=True,
        test__accuracy=acc
    )