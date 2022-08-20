from preprocess import generate_training_seqeunces, SEQUENCE_LENGTH
import keras
import keras.layers

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def build_model(output_unit, num_units, loss, learning_rate):

    # create model architecture
    ## We set the first index of shape as none .i.e = (None, <shape of one unit>) so that we can use how many ever input units we want.
    input = keras.layers.Input(shape=(None, output_unit))
    #The first argument of keras.layers.LSTM is the number of outputs each lstm cell should give out.
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_unit, activation="softmax")(x)
    model = keras.Model(input, output)

    # compile the model
    model.compile(loss=loss,optimizer=keras.optimizers.Adam(lr=learning_rate),metrics=["accuracy"])
    model.summary()
    return model


def train(output_unit = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate = LEARNING_RATE):
    # generate training sequences
    inputs, targets = generate_training_seqeunces(SEQUENCE_LENGTH)

    # build the model
    model = build_model(output_unit, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
