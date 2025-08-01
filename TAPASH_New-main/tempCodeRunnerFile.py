
    conditions = [categories[ans] for ans in ansf]

    if(conditions[0] == conditions[1] and conditions[1] == conditions[2]):
        return conditions[0]