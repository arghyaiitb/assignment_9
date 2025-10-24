#!/usr/bin/env python3
"""
Main CLI Interface for ImageNet-1K Training Pipeline

This script provides a unified command-line interface for:
- Setting up ImageNet dataset access
- Downloading and managing datasets
- Training models
- Checking download status
- Verifying dataset integrity

Usage:
    python main.py setup           # Setup ImageNet access
    python main.py download        # Download dataset
    python main.py train           # Train model
    python main.py status          # Check download status
    python main.py verify          # Verify dataset integrity
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_imagenet_command(args):
    """Run imagenet.py with the given arguments"""
    cmd = [sys.executable, 'imagenet.py'] + args
    return subprocess.run(cmd)


def run_train_command(args):
    """Run train.py with the given arguments"""
    cmd = [sys.executable, 'train.py'] + args
    return subprocess.run(cmd)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="ImageNet-1K Training Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup                    # Setup ImageNet access
  python main.py download --test          # Download test dataset (10k samples)
  python main.py download                 # Download full dataset
  python main.py train --dataset test     # Train on test dataset
  python main.py train --dataset full     # Train on full dataset
  python main.py status                   # Check download status
  python main.py verify                   # Verify dataset integrity

For more options, run:
  python imagenet.py --help
  python train.py --help
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup ImageNet dataset access')
    setup_parser.set_defaults(func=lambda args: run_imagenet_command(['setup']))

    # Download command
    download_parser = subparsers.add_parser('download', help='Download ImageNet dataset')
    download_parser.add_argument('--test', action='store_true', help='Download test dataset')
    download_parser.add_argument('--max-samples', type=int, help='Maximum samples to download')
    download_parser.add_argument('--cache-dir', type=str, help='Cache directory')
    download_parser.set_defaults(func=lambda args: run_imagenet_command([
        'download',
        '--test' if args.test else None,
        f'--max-samples={args.max_samples}' if args.max_samples else None,
        f'--cache-dir={args.cache_dir}' if args.cache_dir else None
    ]))

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ResNet-50 model')
    train_parser.add_argument('--dataset', choices=['test', 'full'], default='test',
                             help='Dataset to use (default: test)')
    train_parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--max-samples', type=int, help='Maximum samples to use')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--cache-dir', type=str, help='Cache directory')
    train_parser.set_defaults(func=lambda args: run_train_command([
        f'--dataset={args.dataset}',
        f'--epochs={args.epochs}',
        f'--batch-size={args.batch_size}' if args.batch_size else None,
        f'--max-samples={args.max_samples}' if args.max_samples else None,
        f'--resume={args.resume}' if args.resume else None,
        f'--cache-dir={args.cache_dir}' if args.cache_dir else None
    ]))

    # Status command
    status_parser = subparsers.add_parser('status', help='Check download status')
    status_parser.add_argument('--test', action='store_true', help='Check test dataset status')
    status_parser.add_argument('--cache-dir', type=str, help='Cache directory')
    status_parser.set_defaults(func=lambda args: run_imagenet_command([
        'status',
        '--test' if args.test else None,
        f'--cache-dir={args.cache_dir}' if args.cache_dir else None
    ]))

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset integrity')
    verify_parser.add_argument('--test', action='store_true', help='Verify test dataset')
    verify_parser.add_argument('--cache-dir', type=str, help='Cache directory')
    verify_parser.set_defaults(func=lambda args: run_imagenet_command([
        'verify',
        '--test' if args.test else None,
        f'--cache-dir={args.cache_dir}' if args.cache_dir else None
    ]))

    # Filter out None values from command arguments
    def filter_args(args_list):
        return [arg for arg in args_list if arg is not None]

    # Override the function to properly handle arguments
    def execute_command(args):
        if args.command == 'setup':
            return run_imagenet_command(['setup'])
        elif args.command == 'download':
            cmd_args = ['download']
            if args.test:
                cmd_args.append('--test')
            if args.max_samples:
                cmd_args.extend(['--max-samples', str(args.max_samples)])
            if args.cache_dir:
                cmd_args.extend(['--cache-dir', args.cache_dir])
            return run_imagenet_command(cmd_args)
        elif args.command == 'train':
            cmd_args = [f'--dataset={args.dataset}', f'--epochs={args.epochs}']
            if args.batch_size:
                cmd_args.extend(['--batch-size', str(args.batch_size)])
            if args.max_samples:
                cmd_args.extend(['--max-samples', str(args.max_samples)])
            if args.resume:
                cmd_args.extend(['--resume', args.resume])
            if args.cache_dir:
                cmd_args.extend(['--cache-dir', args.cache_dir])
            return run_train_command(cmd_args)
        elif args.command == 'status':
            cmd_args = ['status']
            if args.test:
                cmd_args.append('--test')
            if args.cache_dir:
                cmd_args.extend(['--cache-dir', args.cache_dir])
            return run_imagenet_command(cmd_args)
        elif args.command == 'verify':
            cmd_args = ['verify']
            if args.test:
                cmd_args.append('--test')
            if args.cache_dir:
                cmd_args.extend(['--cache-dir', args.cache_dir])
            return run_imagenet_command(cmd_args)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute the command
    result = execute_command(args)

    # Exit with the same code as the subprocess
    sys.exit(result.returncode if result else 0)


if __name__ == "__main__":
    main()
