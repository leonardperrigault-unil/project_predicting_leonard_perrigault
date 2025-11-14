import argparse
from src.data_cleaning import clean

def main():
    parser = argparse.ArgumentParser(
        description='ML Project - World Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-m', '--module',
        type=str,
        required=True,
        choices=['clean','train', 'test'],
        help='Module to run: clean, train, test'
    )

    args = parser.parse_args()

    if args.module == 'clean':
        clean.main(input_file=args.input, output_file=args.output)
    elif args.module == 'train':
        print("t")
    elif args.module == 'test':
        print("t")
    else:
        print(f"Unknown module: {args.module}")

if __name__ == "__main__":
    main()