def arff_to_csv(farff, fcsv):
    data_marker = "@data"
    atrr_marker = "@attribute"
    feature_names = []
    is_data = False

    with open(farff, 'r') as fin:
        with open(fcsv, 'w') as fout:
            for line in fin:
                if is_data:
                    fout.write(line)
                else:
                    if line.startswith(data_marker):
                        is_data = True
                        fout.write(",".join(feature_names) + "\n")
                    else:
                        if line.startswith(atrr_marker):
                            ftrname = line.strip().split()[1]
                            ftrtype = line.strip().split()[2]
                            ftrtype = ftrtype.replace("{", "")
                            ftrtype = ftrtype.replace("}", "")
                            ftrtype = ftrtype.replace(",", ".")
                            feature_names.append("{0}_{1}".format(ftrname, ftrtype))
    print feature_names
