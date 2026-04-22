import argparse
from pathlib import Path

import cover_float.testgen as tg
from cover_float.reference import run_test_vector
from cover_float.scripts.postprocess import postprocess_testvectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input test vector file")
    parser.add_argument("output_file", type=str, help="Path to the output cover vector file")
    parser.add_argument(
        "--suppress-error-check",
        action="store_true",
        help="Suppress error checking between expected and actual results",
    )
    args = parser.parse_args()

    with Path(args.input_file).open("r") as infile, Path(args.output_file).open("w") as outfile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith("//"):
                continue  # Skip empty lines and comments
            result = run_test_vector(line, args.suppress_error_check)
            outfile.write(result)


def auto_parse(model_name: str, output_dir: str) -> None:
    output_directory = Path(output_dir)
    postprocess_testvectors(
        model_name, output_directory / "testvectors", output_directory / "processed", output_directory / "readable"
    )


def testgen() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        action="extend",
        nargs="+",
        help="Model(s) to generate test vectors for",
    )
    parser.add_argument("--output-dir", type=str, default="tests", help="Directory to save generated test vectors")
    args = parser.parse_args()

    if args.models is None:
        tg.B1.main()
        auto_parse("B1", args.output_dir)
        tg.B2.main()
        auto_parse("B2", args.output_dir)
        tg.B3.main()
        auto_parse("B3", args.output_dir)
        tg.B6.main()
        auto_parse("B6", args.output_dir)
        tg.B7.main()
        auto_parse("B7", args.output_dir)
        tg.B8.main()
        auto_parse("B8", args.output_dir)
        tg.B9.main()
        auto_parse("B9", args.output_dir)
        tg.B10.main()
        auto_parse("B10", args.output_dir)
        tg.B11.main()
        auto_parse("B11", args.output_dir)
        tg.B12.main()
        auto_parse("B12", args.output_dir)
        tg.B13.main()
        auto_parse("B13", args.output_dir)
        tg.B14.main()
        auto_parse("B14", args.output_dir)
        tg.B15.main()
        auto_parse("B15", args.output_dir)
        tg.B20.main()
        auto_parse("B20", args.output_dir)
        tg.B21.main()
        auto_parse("B21", args.output_dir)
        tg.B25.main()
        auto_parse("B25", args.output_dir)
        tg.B26.main()
        auto_parse("B26", args.output_dir)
        tg.B27.main()
        auto_parse("B27", args.output_dir)
        tg.B29.main()
        auto_parse("B29", args.output_dir)
    else:
        if "B1" in args.models:
            tg.B1.main()
            auto_parse("B1", args.output_dir)
        if "B2" in args.models:
            tg.B2.main()
            auto_parse("B2", args.output_dir)
        if "B3" in args.models:
            tg.B3.main()
            auto_parse("B3", args.output_dir)
        if "B6" in args.models:
            tg.B6.main()
            auto_parse("B6", args.output_dir)
        if "B7" in args.models:
            tg.B7.main()
            auto_parse("B7", args.output_dir)
        if "B8" in args.models:
            tg.B8.main()
            auto_parse("B8", args.output_dir)
        if "B9" in args.models:
            tg.B9.main()
            auto_parse("B9", args.output_dir)
        if "B10" in args.models:
            tg.B10.main()
            auto_parse("B10", args.output_dir)
        if "B11" in args.models:
            tg.B11.main()
            auto_parse("B11", args.output_dir)
        if "B12" in args.models:
            tg.B12.main()
            auto_parse("B12", args.output_dir)
        if "B13" in args.models:
            tg.B13.main()
            auto_parse("B13", args.output_dir)
        if "B14" in args.models:
            tg.B14.main()
            auto_parse("B14", args.output_dir)
        if "B15" in args.models:
            tg.B15.main()
            auto_parse("B15", args.output_dir)
        if "B20" in args.models:
            tg.B20.main()
            auto_parse("B20", args.output_dir)
        if "B21" in args.models:
            tg.B21.main()
            auto_parse("B21", args.output_dir)
        if "B25" in args.models:
            tg.B25.main()
            auto_parse("B25", args.output_dir)
        if "B26" in args.models:
            tg.B26.main()
            auto_parse("B26", args.output_dir)
        if "B27" in args.models:
            tg.B27.main()
            auto_parse("B27", args.output_dir)
        if "B29" in args.models:
            tg.B29.main()
            auto_parse("B29", args.output_dir)
