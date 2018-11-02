for index, line in enumerate(open('/home/yoma/R/ntu.cc/all_standard.csv', 'r').readlines()):
    w = line.split(',')
    label = w[:1]
    features = w[4:]

    try:
        label = [int(x) for x in label]
        features = [float(x) for x in features]
    except ValueError:
        print('Line %s is corrupt!' % index)
        break
    #
    # if index > 40000:
    #     all1 = label
    #     all1.extend(features)
    #     out2.writerow(all1)
    # else:
    #     all1 = label
    #     all1.extend(features)
    #     out1.writerow(all1)

    X.append(features)
    Y.extend(label)