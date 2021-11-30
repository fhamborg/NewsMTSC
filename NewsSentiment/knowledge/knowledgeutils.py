def find_key_original_or_lc(data: dict, data_keys_lower: dict, key: str):
    # try to get the key as it is from the dict
    if key in data.keys():
        return data[key]
    # if not contained, try whether if using case insensitivity we find an entry
    if key.lower() in data_keys_lower.keys():
        return data_keys_lower[key.lower()]
    # if not, return None
    return set()
