import numpy as np
from sklearn.metrics import accuracy_score
from utils.metrics import precision, recall, f1_score, false_positive_rate
from utils.general_utils import get_logger
from keras.models import model_from_json

logger = get_logger("Evaluate ...")


class Evaluator:
    def __init__(self, model, X_train, X_dev, X_test, Y_train, Y_dev, Y_test, batch_size, save_path):
        self.model = model

        self.batch_size = batch_size

        self.X_train, self.X_dev, self.X_test = X_train, X_dev, X_test
        self.Y_train, self.Y_dev, self.Y_test = Y_train, Y_dev, Y_test

        self.save_path = save_path

        self.best_dev_acc = -1
        self.best_dev_precision = -1
        self.best_dev_recall = -1
        self.best_dev_f1 = -1
        self.best_dev_false_pos_rate = -1

        self.best_test_acc = -1
        self.best_test_precision = -1
        self.best_test_recall = -1
        self.best_test_f1 = -1
        self.best_test_false_pos_rate = -1

        self.dev_acc = None
        self.dev_precision = None
        self.dev_recall = None
        self.dev_f1 = None
        self.dev_false_pos_rate = None

        self.test_acc = None
        self.test_precision = None
        self.test_recall = None
        self.test_f1 = None
        self.test_false_pos_rate = None

    def predict(self):
        train_pred = self.model.predict(self.X_train, batch_size=self.batch_size)
        dev_pred = self.model.predict(self.X_dev, batch_size=self.batch_size)
        test_pred = self.model.predict(self.X_test, batch_size=self.batch_size)

        train_pred = np.round(train_pred)
        dev_pred = np.round(dev_pred)
        test_pred = np.round(test_pred)

        self.dev_acc = accuracy_score(self.Y_dev, dev_pred)
        self.dev_precision = precision(self.Y_dev, dev_pred)
        self.dev_recall = recall(self.Y_dev, dev_pred)
        self.dev_f1 = f1_score(self.Y_dev, dev_pred)
        self.dev_false_pos_rate = false_positive_rate(self.Y_dev, dev_pred)

        self.test_acc = accuracy_score(self.Y_test, test_pred)
        self.test_precision = precision(self.Y_test, test_pred)
        self.test_recall = recall(self.Y_test, test_pred)
        self.test_f1 = f1_score(self.Y_test, test_pred)
        self.test_false_pos_rate = false_positive_rate(self.Y_test, test_pred)

        if self.dev_acc > self.best_dev_acc:
            self.best_dev_acc = self.dev_acc
            self.best_dev_precision = self.dev_precision
            self.best_dev_recall = self.dev_recall
            self.best_dev_f1 = self.dev_f1
            self.best_dev_false_pos_rate = self.dev_false_pos_rate

            self.best_test_acc = self.test_acc
            self.best_test_precision = self.test_precision
            self.best_test_recall = self.test_recall
            self.best_test_f1 = self.test_f1
            self.best_test_false_pos_rate = self.test_false_pos_rate

            # SAVE MODE TO JSON
            model_json = self.model.to_json()
            with open(self.save_path + '.json', 'w') as json_file:
                json_file.write(model_json)
            # SAVE WEIGHTS
            self.model.save_weights(self.save_path + '.h5')
            logger.info("Model saved")

    def print_eval(self):
        logger.info(
            '[DEV]   ACC:  %.3f, PREC: %.3f, REC: %.3f, F1: %.3f, FPR: %.3f \n(Best ACC: {{%.3f}}, Best PREC: {{%.3f}}, Best REC: {{%.3f}}, Best F1: {{%.3f}}, Best FPR: {{%.3f}})' % (
                self.dev_acc, self.dev_precision, self.dev_recall, self.dev_f1, self.dev_false_pos_rate,
                self.best_dev_acc, self.best_dev_precision, self.best_dev_recall, self.best_dev_f1,
                self.best_dev_false_pos_rate))
        logger.info(
            '[TEST]   ACC:  %.3f, PREC: %.3f, REC: %.3f, F1: %.3f, FPR: %.3f \n(Best ACC: {{%.3f}}, Best PREC: {{%.3f}}, Best REC: {{%.3f}}, Best F1: {{%.3f}}, Best FPR: {{%.3f}})' % (
                self.test_acc, self.test_precision, self.test_recall, self.test_f1, self.test_false_pos_rate,
                self.best_test_acc, self.best_test_precision, self.best_test_recall, self.best_test_f1,
                self.best_test_false_pos_rate))
        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_eval(self):
        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')
        logger.info('[DEV]   BEST ACC:  %.3f, BEST PREC: %.3f, BEST REC: %.3f, BEST F1: %.3f, BEST FPR: %.3f' % (
            self.best_dev_acc, self.best_dev_precision, self.best_dev_recall, self.best_dev_f1,
            self.best_dev_false_pos_rate))
        logger.info('[TEST]   BEST ACC:  %.3f, BEST PREC: %.3f, BEST REC: %.3f, BEST F1: %.3f, BEST FPR: %.3f' % (
            self.best_test_acc, self.best_test_precision, self.best_test_recall, self.best_test_f1,
            self.best_test_false_pos_rate))