import numpy as np
from src.models.mnist import load_mnist
from src.models.cnn import CNN

if __name__ == '__main__':
    training_set, training_labels = load_mnist("train", "../data/")
    test_set, test_labels = load_mnist("t10k", "../data/")

    model = CNN()
    count_size = 1000

    loss = 0
    num_correct = 0
    num_epochs = 3

    for epoch in range(num_epochs):
        print('--- Epoch %d ---' % (epoch + 1))

        permute_train = np.random.permutation(len(training_set))
        train_images = training_set[permute_train]
        train_labels = training_labels[permute_train]

        # Train
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            # Print collected statistics every count_size steps
            if i > 0 and i % count_size == count_size - 1:
                print(
                    '[Step %d] Past %d steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, count_size, loss / count_size, num_correct / count_size * 100)
                )
                loss = 0
                num_correct = 0

            l, acc = model.train(im, label)
            loss += l
            num_correct += acc

    # Test the model
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0

    for im, label in zip(test_set, test_labels):
        _, l, acc = model.forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_set)
    print('Test Loss: ', loss / num_tests)
    print('Test Accuracy: ', num_correct / num_tests)
