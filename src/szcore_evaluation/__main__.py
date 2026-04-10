if __name__ == "__main__":
    import argparse

    from szcore_evaluation.evaluate import evaluate_dataset

    parser = argparse.ArgumentParser(
        description="Compare szCORE compliant annotations of EEG datasets of people with epilelpsy."
    )
    parser.add_argument("ref", help="Path to the folder containing the reference TSV files.")
    parser.add_argument("hyp", help="Path to the folder containing the hypothesis TSV files.")
    parser.add_argument("output", help="Path to the output JSON file where the results are saved.")

    args = parser.parse_args()
    evaluate_dataset(args.ref, args.hyp, args.output)