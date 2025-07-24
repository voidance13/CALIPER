import fasttext
# model = fasttext.train_supervised(input="cooking.train")
model = fasttext.train_supervised(input="cooking.train",
                                  autotuneValidationFile='cooking.valid',
                                  autotuneDuration=600,
                                #   autotuneModelSize="2M",
                                  autotuneMetric='f1',
                                  verbose=2)