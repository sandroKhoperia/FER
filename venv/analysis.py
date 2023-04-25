import matplotlib.pyplot as plt




def main():
    labels = ['angry', 'sad', 'happy', 'fear', 'neutral', 'digust', 'surprise']
    values = [958, 1247, 1774, 1024, 1233, 111, 831]
    plt.bar(labels, values)
    plt.title('Emotion Distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Count')

    plt.show()


if __name__ == '__main__':
    main()