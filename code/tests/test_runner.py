"""
Automated Test Runner and Reporting System

Comprehensive test runner that executes all test suites, generates detailed reports,
and provides performance analysis across all advanced training optimizations.
"""

import unittest
import sys
import os
import time
import json
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests import TEST_CONFIG


@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    status: str  # 'PASS', 'FAIL', 'ERROR', 'SKIP'
    duration: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    memory_peak: Optional[float] = None


@dataclass
class TestSuiteResult:
    """Test suite result data structure"""
    name: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    results: List[TestResult]


@dataclass
class TestReport:
    """Complete test report data structure"""
    timestamp: str
    total_duration: float
    total_tests: int
    total_passed: int
    total_failed: int
    total_errors: int
    total_skipped: int
    test_suites: List[TestSuiteResult]
    system_info: Dict[str, Any]
    config: Dict[str, Any]


class TestReporter(unittest.TestResult):
    """Custom test result collector"""

    def __init__(self):
        super().__init__()
        self.test_results = []
        self.current_test_start = None

    def startTest(self, test):
        super().startTest(test)
        self.current_test_start = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.current_test_start
        self.test_results.append(TestResult(
            name=str(test),
            status='PASS',
            duration=duration
        ))

    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.current_test_start
        exc_type, exc_value, exc_traceback = err
        self.test_results.append(TestResult(
            name=str(test),
            status='ERROR',
            duration=duration,
            error_message=str(exc_value),
            error_traceback=''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        ))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.current_test_start
        exc_type, exc_value, exc_traceback = err
        self.test_results.append(TestResult(
            name=str(test),
            status='FAIL',
            duration=duration,
            error_message=str(exc_value),
            error_traceback=''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        ))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.time() - self.current_test_start
        self.test_results.append(TestResult(
            name=str(test),
            status='SKIP',
            duration=duration,
            error_message=reason
        ))


class AdvancedTestRunner:
    """Advanced test runner with comprehensive reporting"""

    def __init__(self, verbosity: int = 2, output_dir: str = None):
        self.verbosity = verbosity
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="llm_test_reports_")
        self.test_modules = [
            'tests.test_advanced_optimizers',
            'tests.test_progressive_training',
            'tests.test_advanced_schedulers',
            'tests.test_fp8_training',
            'tests.test_integration',
            'tests.test_benchmarks',
            'tests.test_stress_edge_cases'
        ]

    def get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        import torch
        import platform
        import psutil

        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

        if torch.cuda.is_available():
            system_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'gpu_memory_gb': [torch.cuda.get_device_properties(i).total_memory / (1024**3)
                                for i in range(torch.cuda.device_count())]
            })

        return system_info

    def run_test_suite(self, module_name: str) -> TestSuiteResult:
        """Run a single test suite"""
        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Import and discover tests
            __import__(module_name)
            module = sys.modules[module_name]
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)

            # Run tests with custom reporter
            reporter = TestReporter()
            suite.run(reporter)

            duration = time.time() - start_time

            # Count results
            passed = len([r for r in reporter.test_results if r.status == 'PASS'])
            failed = len([r for r in reporter.test_results if r.status == 'FAIL'])
            errors = len([r for r in reporter.test_results if r.status == 'ERROR'])
            skipped = len([r for r in reporter.test_results if r.status == 'SKIP'])

            result = TestSuiteResult(
                name=module_name,
                total_tests=len(reporter.test_results),
                passed=passed,
                failed=failed,
                errors=errors,
                skipped=skipped,
                duration=duration,
                results=reporter.test_results
            )

            # Print summary
            print(f"\n{module_name} Summary:")
            print(f"  Tests run: {result.total_tests}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failed}")
            print(f"  Errors: {errors}")
            print(f"  Skipped: {skipped}")
            print(f"  Duration: {duration:.2f}s")

            if failed > 0 or errors > 0:
                print(f"  ❌ FAILED")
            else:
                print(f"  ✅ PASSED")

            return result

        except Exception as e:
            duration = time.time() - start_time
            print(f"Failed to run {module_name}: {e}")
            return TestSuiteResult(
                name=module_name,
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=duration,
                results=[TestResult(
                    name=f"{module_name}.import_error",
                    status='ERROR',
                    duration=duration,
                    error_message=str(e),
                    error_traceback=traceback.format_exc()
                )]
            )

    def run_all_tests(self, selected_suites: List[str] = None) -> TestReport:
        """Run all test suites and generate comprehensive report"""
        print("🚀 Starting Advanced LLM Training Test Suite")
        print(f"Output directory: {self.output_dir}")

        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Select test modules to run
        if selected_suites:
            test_modules = [m for m in self.test_modules if any(suite in m for suite in selected_suites)]
        else:
            test_modules = self.test_modules

        # Run each test suite
        suite_results = []
        for module_name in test_modules:
            try:
                result = self.run_test_suite(module_name)
                suite_results.append(result)
            except KeyboardInterrupt:
                print("\n⚠️  Test execution interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error running {module_name}: {e}")
                continue

        total_duration = time.time() - start_time

        # Aggregate results
        total_tests = sum(s.total_tests for s in suite_results)
        total_passed = sum(s.passed for s in suite_results)
        total_failed = sum(s.failed for s in suite_results)
        total_errors = sum(s.errors for s in suite_results)
        total_skipped = sum(s.skipped for s in suite_results)

        # Create comprehensive report
        report = TestReport(
            timestamp=timestamp,
            total_duration=total_duration,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_errors=total_errors,
            total_skipped=total_skipped,
            test_suites=suite_results,
            system_info=self.get_system_info(),
            config=TEST_CONFIG
        )

        # Generate reports
        self.generate_reports(report)

        # Print final summary
        self.print_final_summary(report)

        return report

    def generate_reports(self, report: TestReport):
        """Generate various report formats"""
        os.makedirs(self.output_dir, exist_ok=True)

        # JSON report
        json_path = os.path.join(self.output_dir, "test_report.json")
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # HTML report
        html_path = os.path.join(self.output_dir, "test_report.html")
        self.generate_html_report(report, html_path)

        # Text summary
        txt_path = os.path.join(self.output_dir, "test_summary.txt")
        self.generate_text_summary(report, txt_path)

        # Performance analysis
        perf_path = os.path.join(self.output_dir, "performance_analysis.json")
        self.generate_performance_analysis(report, perf_path)

        print(f"\n📊 Reports generated:")
        print(f"  📋 Summary: {txt_path}")
        print(f"  🌐 HTML: {html_path}")
        print(f"  📄 JSON: {json_path}")
        print(f"  📈 Performance: {perf_path}")

    def generate_html_report(self, report: TestReport, output_path: str):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced LLM Training Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 10px; background: #e9e9e9; border-radius: 5px; }}
        .passed {{ background: #d4edda; }}
        .failed {{ background: #f8d7da; }}
        .error {{ background: #fff3cd; }}
        .suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .suite-header {{ background: #f8f9fa; padding: 10px; font-weight: bold; }}
        .test-result {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .test-result:last-child {{ border-bottom: none; }}
        .test-pass {{ background: #d4edda; }}
        .test-fail {{ background: #f8d7da; }}
        .test-error {{ background: #fff3cd; }}
        .test-skip {{ background: #e2e3e5; }}
        .error-details {{ font-family: monospace; font-size: 12px; margin-top: 5px; }}
        .system-info {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Advanced LLM Training Test Report</h1>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        <p><strong>Duration:</strong> {report.total_duration:.2f} seconds</p>
        <p><strong>Device:</strong> {report.config.get('device', 'Unknown')}</p>
    </div>

    <div class="summary">
        <div class="stat">
            <h3>{report.total_tests}</h3>
            <p>Total Tests</p>
        </div>
        <div class="stat passed">
            <h3>{report.total_passed}</h3>
            <p>Passed</p>
        </div>
        <div class="stat failed">
            <h3>{report.total_failed}</h3>
            <p>Failed</p>
        </div>
        <div class="stat error">
            <h3>{report.total_errors}</h3>
            <p>Errors</p>
        </div>
        <div class="stat">
            <h3>{report.total_skipped}</h3>
            <p>Skipped</p>
        </div>
    </div>

    <div class="system-info">
        <h3>System Information</h3>
        <p><strong>Platform:</strong> {report.system_info.get('platform', 'Unknown')}</p>
        <p><strong>Python:</strong> {report.system_info.get('python_version', 'Unknown')}</p>
        <p><strong>PyTorch:</strong> {report.system_info.get('pytorch_version', 'Unknown')}</p>
        <p><strong>CUDA Available:</strong> {report.system_info.get('cuda_available', False)}</p>
        <p><strong>CPU Count:</strong> {report.system_info.get('cpu_count', 'Unknown')}</p>
        <p><strong>Memory:</strong> {report.system_info.get('memory_total_gb', 0):.1f} GB</p>
    </div>

    <h2>Test Suite Results</h2>
"""

        # Add test suite details
        for suite in report.test_suites:
            suite_status = "✅ PASSED" if (suite.failed + suite.errors) == 0 else "❌ FAILED"
            html_content += f"""
    <div class="suite">
        <div class="suite-header">
            {suite.name} - {suite_status}
            <small>({suite.passed}/{suite.total_tests} passed, {suite.duration:.2f}s)</small>
        </div>
"""

            for test in suite.results:
                status_class = f"test-{test.status.lower()}"
                status_icon = {
                    'PASS': '✅',
                    'FAIL': '❌',
                    'ERROR': '⚠️',
                    'SKIP': '⏭️'
                }.get(test.status, '?')

                html_content += f"""
        <div class="test-result {status_class}">
            <div>
                {status_icon} {test.name}
                <small>({test.duration:.3f}s)</small>
            </div>
"""

                if test.error_message:
                    html_content += f"""
            <div class="error-details">
                <strong>Error:</strong> {test.error_message}
            </div>
"""

                html_content += "</div>"

            html_content += "</div>"

        html_content += """
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html_content)

    def generate_text_summary(self, report: TestReport, output_path: str):
        """Generate text summary report"""
        content = f"""
Advanced LLM Training Test Suite - Summary Report
================================================

Timestamp: {report.timestamp}
Total Duration: {report.total_duration:.2f} seconds

Overall Results:
  Total Tests: {report.total_tests}
  Passed: {report.total_passed}
  Failed: {report.total_failed}
  Errors: {report.total_errors}
  Skipped: {report.total_skipped}
  Success Rate: {(report.total_passed / max(report.total_tests, 1)) * 100:.1f}%

System Information:
  Platform: {report.system_info.get('platform', 'Unknown')}
  Python Version: {report.system_info.get('python_version', 'Unknown')}
  PyTorch Version: {report.system_info.get('pytorch_version', 'Unknown')}
  CUDA Available: {report.system_info.get('cuda_available', False)}
  Device: {report.config.get('device', 'Unknown')}

Test Suite Results:
==================
"""

        for suite in report.test_suites:
            status = "PASSED" if (suite.failed + suite.errors) == 0 else "FAILED"
            content += f"""
{suite.name}: {status}
  Tests: {suite.total_tests}
  Passed: {suite.passed}
  Failed: {suite.failed}
  Errors: {suite.errors}
  Skipped: {suite.skipped}
  Duration: {suite.duration:.2f}s
"""

            # Add failed/error test details
            failed_tests = [t for t in suite.results if t.status in ['FAIL', 'ERROR']]
            if failed_tests:
                content += "  Failed/Error Tests:\n"
                for test in failed_tests:
                    content += f"    - {test.name}: {test.error_message}\n"

        with open(output_path, 'w') as f:
            f.write(content)

    def generate_performance_analysis(self, report: TestReport, output_path: str):
        """Generate performance analysis"""
        analysis = {
            'timestamp': report.timestamp,
            'total_duration': report.total_duration,
            'suite_performance': [],
            'slowest_tests': [],
            'test_categories': {}
        }

        # Suite performance
        for suite in report.test_suites:
            analysis['suite_performance'].append({
                'name': suite.name,
                'duration': suite.duration,
                'test_count': suite.total_tests,
                'avg_test_time': suite.duration / max(suite.total_tests, 1),
                'success_rate': suite.passed / max(suite.total_tests, 1)
            })

        # Slowest tests
        all_tests = []
        for suite in report.test_suites:
            all_tests.extend(suite.results)

        slowest_tests = sorted(all_tests, key=lambda x: x.duration, reverse=True)[:10]
        analysis['slowest_tests'] = [
            {
                'name': test.name,
                'duration': test.duration,
                'status': test.status
            }
            for test in slowest_tests
        ]

        # Test categories
        categories = {
            'optimizer': 0,
            'scheduler': 0,
            'progressive': 0,
            'fp8': 0,
            'integration': 0,
            'benchmark': 0,
            'stress': 0
        }

        for test in all_tests:
            test_name = test.name.lower()
            for category in categories:
                if category in test_name:
                    categories[category] += 1
                    break

        analysis['test_categories'] = categories

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

    def print_final_summary(self, report: TestReport):
        """Print final test summary"""
        print(f"\n{'='*80}")
        print("🎯 FINAL TEST SUMMARY")
        print(f"{'='*80}")

        # Overall status
        if report.total_failed == 0 and report.total_errors == 0:
            status = "✅ ALL TESTS PASSED"
            status_color = "\033[92m"  # Green
        else:
            status = "❌ SOME TESTS FAILED"
            status_color = "\033[91m"  # Red

        print(f"{status_color}{status}\033[0m")

        print(f"\n📊 Results:")
        print(f"  Total Tests: {report.total_tests}")
        print(f"  Passed: {report.total_passed}")
        print(f"  Failed: {report.total_failed}")
        print(f"  Errors: {report.total_errors}")
        print(f"  Skipped: {report.total_skipped}")
        print(f"  Success Rate: {(report.total_passed / max(report.total_tests, 1)) * 100:.1f}%")
        print(f"  Total Duration: {report.total_duration:.2f} seconds")

        # Suite breakdown
        print(f"\n📋 Suite Breakdown:")
        for suite in report.test_suites:
            suite_status = "✅" if (suite.failed + suite.errors) == 0 else "❌"
            print(f"  {suite_status} {suite.name}: "
                  f"{suite.passed}/{suite.total_tests} passed ({suite.duration:.1f}s)")

        # Failed tests summary
        failed_tests = []
        for suite in report.test_suites:
            failed_tests.extend([t for t in suite.results if t.status in ['FAIL', 'ERROR']])

        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests[:10]:  # Show first 10
                print(f"  - {test.name}: {test.error_message}")
            if len(failed_tests) > 10:
                print(f"  ... and {len(failed_tests) - 10} more")

        print(f"\n📁 Reports saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="Advanced LLM Training Test Runner")
    parser.add_argument(
        '--suites',
        nargs='*',
        help='Specific test suites to run (optimizers, progressive, schedulers, fp8, integration, benchmarks, stress)',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for reports',
        default=None
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=2,
        help='Test verbosity level (0-2)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test suite (skip benchmarks and stress tests)'
    )

    args = parser.parse_args()

    # Determine test suites
    if args.quick:
        selected_suites = ['optimizers', 'progressive', 'schedulers', 'fp8', 'integration']
    else:
        selected_suites = args.suites

    # Create test runner
    runner = AdvancedTestRunner(
        verbosity=args.verbosity,
        output_dir=args.output_dir
    )

    try:
        # Run tests
        report = runner.run_all_tests(selected_suites)

        # Exit with appropriate code
        if report.total_failed > 0 or report.total_errors > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()