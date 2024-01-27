class IntentClassifier:
    def __init__(self):
        print('IntentClassifier::init')

    def classify_intent(self, input_msg):
        if self:
            print("input_msg", input_msg)
            return input_msg


