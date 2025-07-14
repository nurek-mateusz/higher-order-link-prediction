import os

# coauth-DBLP: years (e.g., 1974, 1981)
# coauth-MAG-Geology: years (e.g., 1987, 2007)
# coauth-MAG-History: years (e.g., 2004, 2016)
# congress-bills: days (e.g., 720261, 720288)
# contact-high-school: 20 seconds interval (e.g., 1385982020)
# contact-primary-school: 20 seconds interval (e.g., 31220)
# DAWN: quarter-years, year * 4 + quarter, where year ranges from 2004 to 2011 and quarter ranges from 1 to 4 (e.g, 8019)
# email-Enron: milliseconds (e.g., 63083183340000)
# email-Eu: 1 second resolution (e.g., 1064021027)
# NDC-classes: days (expressed in milliseconds), (e.g., 63474192000000)
# NDC-substances: days (expressed in milliseconds), (e.g., 63474192000000)
# tags-ask-ubuntu: milliseconds (e.g., 48942985907)
# tags-math-sx: milliseconds (e.g., 9952566473)
# threads-ask-ubuntu: milliseconds (e.g., 136124026250)
# threads-math-sx: milliseconds (e.g., 74596128084)
# threads-stack-overflow: milliseconds (e.g., 227282834507)

dataset_names = [
    'coauth-DBLP',
    'coauth-MAG-Geology',
    'coauth-MAG-History',
    'congress-bills',
    'contact-high-school',
    'contact-primary-school',
    'DAWN',
    'email-Enron',
    'email-Eu',
    'NDC-classes',
    'NDC-substances',
    'tags-ask-ubuntu',
    'tags-math-sx',
    'tags-stack-overflow',
    'threads-ask-ubuntu',
    'threads-math-sx',
    'threads-stack-overflow'
]

for dataset in dataset_names:
    # ScHoLP format:
    # Consider a dataset consisting of three simplices:
    #     1. {1, 2, 3} at time 10
    #     2. {2, 4} at time 15.
    #     3. {1, 3, 4, 5} at time 21.
    # Then the data structure would be:
    #     - simplices = [1, 2, 3, 2, 4, 1, 3, 4, 5]
    #     - nverts = [3, 2, 4]
    #     - times = [10, 15, 21]
    
    # Read data
    with open(f'data/raw/{dataset}/{dataset}-simplices.txt', 'r') as file:
        verts = [int(line.strip()) for line in file]

    with open(f'data/raw/{dataset}/{dataset}-nverts.txt', 'r') as file:
        nverts = [int(line.strip()) for line in file]

    with open(f'data/raw/{dataset}/{dataset}-times.txt', 'r') as file:
        times = [int(line.strip()) for line in file]

    # Create simplices
    simplices = []
    index = 0
    for count in nverts:
        simplex = verts[index:index+count]
        simplices.append(simplex)
        index += count

    # Combine simplices with their timestamps and sort them chronologically
    combined = list(zip(simplices, nverts, times))
    combined.sort(key=lambda x: x[2])

    # Remove simplices with fewer than two vertices
    combined = [x for x in combined if x[1] >= 2]

    # Split the data into training and test sets based on time
    min_time = combined[0][2]
    max_time = combined[-1][2]
    threshold = min_time + round((max_time - min_time) * 0.8)

    train = [x for x in combined if x[2] <= threshold]
    test = [x for x in combined if x[2] > threshold]

    # Remove from the test set all simplices that appear in the train set.
    # Using frozenset allows comparing simplices as unordered, hashable sets of vertices.
    train_simplex_sets = set(frozenset(s) for s, _, _ in train)

    # Use a set to keep only unique simplices from the test set.
    # We care about which simplices appeared, not how many times they occurred.
    test_simplex_sets = set(
        frozenset(s) for s, _, _ in test
        if frozenset(s) not in train_simplex_sets
    )
    test = test_simplex_sets

    # Save train and test sets in ScHoLP format
    def unpack_train(data):
        verts = []
        nverts = []
        times = []
        for v, n, t in data:
            verts.extend(v)
            nverts.append(n)
            times.append(t)
        return verts, nverts, times
    
    def unpack_test(data):
        verts = []
        nverts = []
        for v in data:
            verts.extend(v)
            nverts.append(len(v))
        return verts, nverts

    train_verts, train_nverts, train_times = unpack_train(train)
    test_verts, test_nverts = unpack_test(test)

    def save(filepath, data):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for item in data:
                f.write(f"{item}\n")

    save(f'data/train/{dataset}/{dataset}-simplices.txt', train_verts)
    save(f'data/train/{dataset}/{dataset}-nverts.txt', train_nverts)
    save(f'data/train/{dataset}/{dataset}-times.txt', train_times)

    save(f'data/test/{dataset}/{dataset}-simplices.txt', test_verts)
    save(f'data/test/{dataset}/{dataset}-nverts.txt', test_nverts)