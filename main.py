#!/usr/bin/env python
"""
SPY Trading Strategies Launcher
Main script to run multiple SPY trading strategies in parallel from YAML configuration.
"""

import yaml
import subprocess
import threading
import time
import signal
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional


class StrategyManager:
    """Manages multiple trading strategy processes."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = True
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.config_file}' not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('global', {}).get('log_level', 'INFO')
        
        # Create handlers with explicit UTF-8 encoding
        file_handler = logging.FileHandler('strategy_manager.log', encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)
        
        # Set the same formatter for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        # Configure logging with explicit handlers
        logging.basicConfig(
            level=getattr(logging, log_level),
            handlers=[file_handler, stream_handler]
        )
        self.logger = logging.getLogger('StrategyManager')
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.running = False
        self.stop_all_strategies()
    
    def _build_command(self, strategy_name: str, strategy_config: dict) -> List[str]:
        """Build command line arguments for a strategy."""
        script = strategy_config['script']
        args = strategy_config.get('args', {})
        
        # Verify script exists
        if not Path(script).exists():
            raise FileNotFoundError(f"Strategy script '{script}' not found.")
        
        # Build command - use the bundled Python executable
        cmd = [sys.executable, script]
        
        # Add arguments
        for key, value in args.items():
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def _run_strategy(self, strategy_name: str, strategy_config: dict):
        """Run a single strategy in a subprocess with monitoring."""
        max_retries = self.config.get('global', {}).get('max_retries', 3)
        restart_on_failure = self.config.get('global', {}).get('restart_on_failure', True)
        
        retry_count = 0
        
        while self.running and retry_count <= max_retries:
            try:
                cmd = self._build_command(strategy_name, strategy_config)
                self.logger.info(f"Starting {strategy_name} strategy: {' '.join(cmd)}")
                
                # Set up environment for bundled app
                env = os.environ.copy()
                
                # Force UTF-8 encoding in subprocess environment
                env['PYTHONIOENCODING'] = 'utf-8'
                env['LC_ALL'] = 'en_US.UTF-8'
                env['LANG'] = 'en_US.UTF-8'
                
                # Always set PYTHONPATH to match the main app's sys.path
                # This ensures subprocess has same module search paths as main app
                current_pythonpath = env.get('PYTHONPATH', '')
                main_app_paths = os.pathsep.join(sys.path)
                
                if current_pythonpath:
                    env['PYTHONPATH'] = f"{main_app_paths}{os.pathsep}{current_pythonpath}"
                else:
                    env['PYTHONPATH'] = main_app_paths
                
                self.logger.info(f"Set subprocess PYTHONPATH to match main app sys.path")
                self.logger.debug(f"PYTHONPATH: {env['PYTHONPATH']}")
                
                # For macOS app bundles, add additional bundle-specific setup if needed
                if getattr(sys, 'frozen', False):
                    self.logger.info("Bundle detected - using bundled Python with system paths")
                else:
                    self.logger.info("Development mode - using system Python environment")
                
                # Start the process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env
                )
                
                self.processes[strategy_name] = process
                
                # Monitor process output
                for line in iter(process.stdout.readline, ''):
                    if not self.running:
                        break
                    if line.strip():
                        self.logger.info(f"[{strategy_name}] {line.strip()}")
                
                # Wait for process to complete
                return_code = process.wait()
                
                if return_code == 0:
                    self.logger.info(f"{strategy_name} strategy completed successfully.")
                    break
                else:
                    self.logger.error(f"{strategy_name} strategy exited with code {return_code}")
                    
                    if restart_on_failure and retry_count < max_retries:
                        retry_count += 1
                        self.logger.info(f"Restarting {strategy_name} strategy (attempt {retry_count}/{max_retries})")
                        time.sleep(5)  # Wait before restart
                    else:
                        break
                        
            except FileNotFoundError as e:
                self.logger.error(f"Could not start {strategy_name}: {e}")
                break
            except Exception as e:
                self.logger.error(f"Error running {strategy_name}: {e}")
                if not restart_on_failure:
                    break
                retry_count += 1
                if retry_count <= max_retries:
                    self.logger.info(f"Retrying {strategy_name} (attempt {retry_count}/{max_retries})")
                    time.sleep(5)
        
        # Clean up
        if strategy_name in self.processes:
            del self.processes[strategy_name]
        self.logger.info(f"{strategy_name} strategy thread ending.")
    
    def start_strategies(self):
        """Start all enabled strategies."""
        strategies = self.config.get('strategies', {})
        
        if not strategies:
            self.logger.error("No strategies configured.")
            return
        
        enabled_strategies = {name: config for name, config in strategies.items() 
                            if config.get('enabled', False)}
        
        if not enabled_strategies:
            self.logger.error("No strategies enabled.")
            return
        
        self.logger.info(f"Starting {len(enabled_strategies)} strategies...")
        
        # Start each strategy in its own thread
        for strategy_name, strategy_config in enabled_strategies.items():
            thread = threading.Thread(
                target=self._run_strategy,
                args=(strategy_name, strategy_config),
                name=f"Strategy-{strategy_name}",
                daemon=False
            )
            self.threads[strategy_name] = thread
            thread.start()
        
        self.logger.info("All strategies started.")
    
    def stop_all_strategies(self):
        """Stop all running strategies."""
        self.logger.info("Stopping all strategies...")
        
        # Terminate processes
        for strategy_name, process in self.processes.items():
            try:
                self.logger.info(f"Terminating {strategy_name}...")
                process.terminate()
                
                # Wait for graceful shutdown, then force kill if needed
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {strategy_name}...")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                self.logger.error(f"Error stopping {strategy_name}: {e}")
        
        # Wait for all threads to complete
        for strategy_name, thread in self.threads.items():
            try:
                thread.join(timeout=5)
                if thread.is_alive():
                    self.logger.warning(f"Thread {strategy_name} did not exit cleanly")
            except Exception as e:
                self.logger.error(f"Error joining thread {strategy_name}: {e}")
        
        self.logger.info("All strategies stopped.")
    
    def wait_for_completion(self):
        """Wait for all strategy threads to complete."""
        try:
            while self.running and any(thread.is_alive() for thread in self.threads.values()):
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received.")
            self.running = False
    
    def run(self):
        """Main execution method."""
        self.logger.info("SPY Trading Strategies Manager starting...")
        
        # Test pandas import in main app
        try:
            import pandas as pd
            self.logger.info(f"SUCCESS: Main app can import pandas from: {pd.__file__}")
        except ImportError as e:
            self.logger.error(f"ERROR: Main app cannot import pandas: {e}")
        
        # Test ib_insync import in main app
        try:
            import ib_insync
            self.logger.info(f"SUCCESS: Main app can import ib_insync from: {ib_insync.__file__}")
        except ImportError as e:
            self.logger.error(f"ERROR: Main app cannot import ib_insync: {e}")
        
        try:
            self.start_strategies()
            self.wait_for_completion()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            if self.running:
                self.stop_all_strategies()
            self.logger.info("SPY Trading Strategies Manager shutdown complete.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPY Trading Strategies Manager")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        try:
            manager = StrategyManager(args.config)
            print("Configuration validation successful.")
            
            # Print enabled strategies
            strategies = manager.config.get('strategies', {})
            enabled = [name for name, config in strategies.items() if config.get('enabled', False)]
            print(f"Enabled strategies: {', '.join(enabled) if enabled else 'None'}")
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            sys.exit(1)
    else:
        manager = StrategyManager(args.config)
        manager.run()


if __name__ == "__main__":
    main() 
