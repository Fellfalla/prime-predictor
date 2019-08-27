import tensorflow as tf
import numpy as np

prime_states = {
    2 : True,
    3 : True,
    4 : False
}

def is_prime(givenNumber):  
    if givenNumber not in prime_states:
        prime_states[givenNumber] = True
        for num in range(2, int(givenNumber ** 0.5) + 1):
            if givenNumber % num == 0:
                prime_states[givenNumber] = False
                break
    
    return prime_states[givenNumber]

def get_next_prime(x):
    while not is_prime(x):
        x = x+1
    return x

if __name__ == "__main__":
    a = get_next_prime(500)
    print(a)

    prime_model = tf.keras.Sequential()
    prime_model.add(tf.keras.layers.Dense(24, activation='relu', input_dim=1))
    prime_model.add(tf.keras.layers.Dense(24, activation='relu'))
    prime_model.add(tf.keras.layers.Dense(24, activation='relu'))
    prime_model.add(tf.keras.layers.Dense(24, activation='relu'))
    prime_model.add(tf.keras.layers.Dense(24, activation='relu'))
    prime_model.add(tf.keras.layers.Dense(1))

    opt = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.mae

    prime_model.compile(optimizer=opt, loss=loss)
    prime_model.summary()

    batch_size = 64
    steps = 500
    ##### Run the Training #####
    for i in range(steps):
        x = np.array(range(i*batch_size))
        y = np.array([get_next_prime(n) for n in x])
        r = prime_model.train_on_batch(x, y)
        print(r)
        # print(r.history['loss'])

    ##### Run the Evaluation #####
    for i in range(steps*batch_size,steps*batch_size+50):
        x = np.array(i).reshape((1,1))
        
        expected = get_next_prime(i)
        pred = prime_model.predict(x)

        print("input: %i\texpected: %i\tpredicted %i"%(i, expected, pred[0]))


