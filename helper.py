def encode_nominal_test_attributes(data_frame, columns):
    y = data_frame[columns]
    data = []
    for index, row in y.iterrows():
        b = []
        if row['A1'] == 'a':
            a1 = [1, 0]
        else:
            a1 = [0, 1]

        if row['A3'] == 'l':
            a3 = [1, 0, 0]
        elif row['A3'] == 'u':
            a3 = [0, 1, 0]
        else:
            a3 = [0, 0, 1]

        if row['A4'] == 'g':
            a4 = [1, 0, 0]
        elif row['A4'] == 'gg':
            a4 = [0, 1, 0]
        else:
            a4 = [0, 0, 1]

        if row['A6'] == 'aa':
            a6 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'c':
            a6 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'cc':
            a6 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'd':
            a6 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'e':
            a6 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'ff':
            a6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'i':
            a6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'j':
            a6 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif row['A6'] == 'k':
            a6 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif row['A6'] == 'm':
            a6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif row['A6'] == 'q':
            a6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif row['A6'] == 'r':
            a6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif row['A6'] == 'w':
            a6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        else:
            a6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        if row['A8'] == False:
            a8 = [1, 0]
        else:
            a8 = [0, 1]

        if row['A9'] == 'bb':
            a9 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif row['A9'] == 'dd':
            a9 = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif row['A9'] == 'ff':
            a9 = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif row['A9'] == 'h':
            a9 = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif row['A9'] == 'j':
            a9 = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif row['A9'] == 'n':
            a9 = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif row['A9'] == 'o':
            a9 = [0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif row['A9'] == 'v':
            a9 = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        else:
            a9 = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        if row['A11'] == False:
            a11 = [1, 0]
        else:
            a11 = [0, 1]

        if row['A13'] == False:
            a13 = [1, 0]
        else:
            a13 = [0, 1]

        b = np.concatenate([a1, a3, a4, a6, a8, a9, a11, a13])
        data.append(b)

    n = pd.DataFrame(data)
    n.columns = ['a', 'b', 'l', 'u', 'y', 'g', 'gg', 'p', 'aa', 'c', 'cc', 'd', 'e', 'ff', 'i', 'j', 'k', 'm', 'q', 'r',
                 'w', 'x', 'a8_False', 'a8_true', 'bb', 'dd', 'ff', 'h', 'j', 'n', 'o', 'v', 'z', 'a11_False',
                 'a11_true', 'a13_False', 'a13_true']
    return n

