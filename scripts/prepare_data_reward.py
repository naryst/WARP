import datasets
import argparse

def prepare_data(split):
    imdb_data = datasets.load_dataset("stanfordnlp/imdb")[split]

    total_samples = len(imdb_data)
    half = total_samples // 2
    # first half of the data - negative comments, the rest - positive
    neg_comm = imdb_data.select(range(half))
    pos_comm = imdb_data.select(range(half, total_samples))

    # data consistency check
    assert sum(neg_comm["label"]) == 0
    assert sum(pos_comm["label"]) == len(pos_comm)

    pairs_dataset = datasets.Dataset.from_dict(
        {
            "negative_comment": neg_comm["text"],
            "positive_comment": pos_comm["text"],
        }
    )

    pairs_dataset.save_to_disk("comment_pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IMDb data.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (train, test, validation)")
    args = parser.parse_args()
    prepare_data(args.split)
