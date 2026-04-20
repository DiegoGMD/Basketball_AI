"""
🏀 BASKETBALL DETECTION - MODEL PERFORMANCE METRICS
=============================================================
This script analyzes and explains the performance of trained YOLO models.
Generates visualizations, statistics, and detailed performance reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class MetricsConfig:
    PROJECT_NAME = "basketball_training"
    RUN_NAME = "yolo11s_5classes"
    MODEL_PATH = Path(f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    RESULTS_CSV = Path(f"{PROJECT_NAME}/{RUN_NAME}/results.csv")
    DATASET_DIR = Path("basketball-detection-srfkd-1")
    DATA_YAML = DATASET_DIR / "data_basketball.yaml"
    OUTPUT_DIR = Path("metrics_reports")
    
# Create output directory
MetricsConfig.OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================================================================
# 1. TRAINING HISTORY ANALYSIS
# ==============================================================================
class TrainingAnalyzer:
    """Analyzes training history and performance trends."""
    
    def __init__(self, results_csv):
        self.csv_path = results_csv
        if not self.csv_path.exists():
            print(f"❌ Results CSV not found at {self.csv_path}")
            self.df = None
            return
        
        self.df = pd.read_csv(results_csv)
        self.df.columns = self.df.columns.str.strip()  # Remove whitespace from column names
        
    def get_summary(self):
        """Get training summary statistics."""
        if self.df is None:
            return None
            
        summary = {
            'total_epochs': len(self.df),
            'total_time_hours': self.df['time'].sum() / 3600,
            'avg_epoch_time_min': self.df['time'].mean() / 60,
            'best_map50': self.df['metrics/mAP50(B)'].max(),
            'best_map5095': self.df['metrics/mAP50-95(B)'].max(),
            'best_precision': self.df['metrics/precision(B)'].max(),
            'best_recall': self.df['metrics/recall(B)'].max(),
            'final_train_box_loss': self.df['train/box_loss'].iloc[-1],
            'final_val_box_loss': self.df['val/box_loss'].iloc[-1],
            'best_epoch': self.df['metrics/mAP50-95(B)'].idxmax() + 1,
        }
        return summary
    
    def print_summary(self):
        """Print formatted training summary."""
        summary = self.get_summary()
        if summary is None:
            return
            
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        print(f"Total Epochs:           {summary['total_epochs']}")
        print(f"Total Training Time:    {summary['total_time_hours']:.1f} hours")
        print(f"Avg Time per Epoch:     {summary['avg_epoch_time_min']:.1f} min")
        print(f"Best Epoch:             {summary['best_epoch']}")
        print()
        print("🎯 BEST METRICS ACHIEVED:")
        print(f"   • mAP@50:             {summary['best_map50']:.4f}")
        print(f"   • mAP@50-95:          {summary['best_map5095']:.4f}")
        print(f"   • Precision:          {summary['best_precision']:.4f}")
        print(f"   • Recall:             {summary['best_recall']:.4f}")
        print()
        print("📉 FINAL LOSSES:")
        print(f"   • Train Box Loss:     {summary['final_train_box_loss']:.4f}")
        print(f"   • Val Box Loss:       {summary['final_val_box_loss']:.4f}")
        print("="*70 + "\n")
        
        return summary
    
    def plot_training_curves(self):
        """Generate training loss and metrics curves."""
        if self.df is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History & Performance Metrics', fontsize=16, fontweight='bold')
        
        # 1. Loss Curves
        ax = axes[0, 0]
        ax.plot(self.df.index + 1, self.df['train/box_loss'], label='Train Box Loss', linewidth=2)
        ax.plot(self.df.index + 1, self.df['val/box_loss'], label='Val Box Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Box Loss Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. mAP Curves
        ax = axes[0, 1]
        ax.plot(self.df.index + 1, self.df['metrics/mAP50(B)'], label='mAP@50', linewidth=2, marker='o', markersize=3)
        ax.plot(self.df.index + 1, self.df['metrics/mAP50-95(B)'], label='mAP@50-95', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Precision & Recall
        ax = axes[1, 0]
        ax.plot(self.df.index + 1, self.df['metrics/precision(B)'], label='Precision', linewidth=2, marker='o', markersize=3)
        ax.plot(self.df.index + 1, self.df['metrics/recall(B)'], label='Recall', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision vs Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 4. Classification Loss
        ax = axes[1, 1]
        ax.plot(self.df.index + 1, self.df['train/cls_loss'], label='Train Class Loss', linewidth=2)
        ax.plot(self.df.index + 1, self.df['val/cls_loss'], label='Val Class Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = MetricsConfig.OUTPUT_DIR / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_overfitting_analysis(self):
        """Analyze overfitting patterns."""
        if self.df is None:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Overfitting Analysis', fontsize=16, fontweight='bold')
        
        # Train vs Val Loss Gap
        ax = axes[0]
        train_loss = self.df['train/box_loss'].values
        val_loss = self.df['val/box_loss'].values
        gap = val_loss - train_loss
        
        ax.fill_between(self.df.index + 1, train_loss, val_loss, alpha=0.3, label='Train-Val Gap')
        ax.plot(self.df.index + 1, train_loss, label='Train Loss', linewidth=2)
        ax.plot(self.df.index + 1, val_loss, label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Box Loss')
        ax.set_title('Train vs Validation Loss Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overfitting Score
        ax = axes[1]
        overfitting_score = (gap / train_loss * 100).clip(0, 100)
        colors = ['green' if x < 10 else 'orange' if x < 20 else 'red' for x in overfitting_score]
        ax.bar(self.df.index + 1, overfitting_score, color=colors, alpha=0.7)
        ax.axhline(y=10, color='g', linestyle='--', label='Good', alpha=0.5)
        ax.axhline(y=20, color='orange', linestyle='--', label='Moderate', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Overfitting Score (%)')
        ax.set_title('Overfitting Progression')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = MetricsConfig.OUTPUT_DIR / 'overfitting_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

# ==============================================================================
# 2. MODEL VALIDATION & TEST SET EVALUATION
# ==============================================================================
class ModelEvaluator:
    """Evaluates model on test set and generates detailed metrics."""
    
    def __init__(self, model_path, dataset_yaml):
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            self.model = None
            return
        
        self.model = YOLO(str(model_path))
        self.dataset_yaml = dataset_yaml
        self.results = None
    
    def load_class_names(self):
        """Load class names from data.yaml."""
        if not Path(self.dataset_yaml).exists():
            print(f"❌ Dataset YAML not found at {self.dataset_yaml}")
            return None
        
        with open(self.dataset_yaml, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', {})
    
    def validate(self):
        """Run validation on test set."""
        if self.model is None:
            return None
        
        print("\n🔍 Running validation on test set...")
        self.results = self.model.val(data=str(self.dataset_yaml), verbose=False)
        return self.results
    
    def print_validation_report(self):
        """Print detailed validation report."""
        if self.results is None:
            print("❌ No validation results available. Run validate() first.")
            return
        
        print("\n" + "="*70)
        print("✅ VALIDATION RESULTS")
        print("="*70)
        
        class_names = self.load_class_names()
        
        # Overall metrics
        print(f"\n📊 Overall Performance:")
        print(f"   • mAP@50:             {self.results.box.map50:.4f}")
        print(f"   • mAP@50-95:          {self.results.box.map:.4f}")
        print(f"   • Precision:          {self.results.box.mp:.4f}")
        print(f"   • Recall:             {self.results.box.mr:.4f}")
        
        # Per-class metrics
        if hasattr(self.results.box, 'map50_per_class'):
            print(f"\n🏀 Per-Class Performance:")
            print(f"{'Class':<20} {'mAP@50':<12} {'Precision':<12} {'Recall':<12}")
            print("-" * 56)
            
            for cls_idx, cls_name in class_names.items():
                map50 = self.results.box.map50_per_class[cls_idx] if hasattr(self.results.box, 'map50_per_class') else 0
                prec = self.results.box.p_per_class[cls_idx] if hasattr(self.results.box, 'p_per_class') else 0
                rec = self.results.box.r_per_class[cls_idx] if hasattr(self.results.box, 'r_per_class') else 0
                print(f"{cls_name:<20} {map50:<12.4f} {prec:<12.4f} {rec:<12.4f}")
        
        print("="*70 + "\n")
    
    def plot_validation_results(self):
        """Plot validation metrics."""
        if self.results is None:
            return
        
        class_names = self.load_class_names()
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Validation Results by Class', fontsize=16, fontweight='bold')
        
        class_list = [class_names[i] for i in range(len(class_names))]
        
        # mAP50 per class
        if hasattr(self.results.box, 'map50_per_class'):
            ax = axes[0]
            map50_vals = self.results.box.map50_per_class
            colors = plt.cm.viridis(np.linspace(0, 1, len(map50_vals)))
            ax.bar(class_list, map50_vals, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('mAP@50')
            ax.set_title('mAP@50 per Class')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Precision per class
        if hasattr(self.results.box, 'p_per_class'):
            ax = axes[1]
            prec_vals = self.results.box.p_per_class
            colors = plt.cm.plasma(np.linspace(0, 1, len(prec_vals)))
            ax.bar(class_list, prec_vals, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Precision')
            ax.set_title('Precision per Class')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Recall per class
        if hasattr(self.results.box, 'r_per_class'):
            ax = axes[2]
            rec_vals = self.results.box.r_per_class
            colors = plt.cm.cool(np.linspace(0, 1, len(rec_vals)))
            ax.bar(class_list, rec_vals, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Recall')
            ax.set_title('Recall per Class')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # It is not detecting the map50 per class

        # plt.tight_layout()
        # save_path = MetricsConfig.OUTPUT_DIR / 'validation_results.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"✅ Saved: {save_path}")
        plt.close()

# ==============================================================================
# 3. PERFORMANCE INSIGHTS & EXPLANATIONS
# ==============================================================================
class PerformanceInsights:
    """Generates human-readable performance explanations."""
    
    def __init__(self, training_summary, validation_results=None):
        self.training_summary = training_summary
        self.validation_results = validation_results
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        if self.training_summary is None:
            return None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': MetricsConfig.MODEL_PATH.as_posix(),
            'summary': self.generate_summary(),
            'strengths': self.analyze_strengths(),
            'weaknesses': self.analyze_weaknesses(),
            'recommendations': self.generate_recommendations(),
        }
        return report
    
    def generate_summary(self):
        """Generate training summary text."""
        s = self.training_summary
        
        summary = f"""
Training completed in {s['total_time_hours']:.1f} hours ({s['total_epochs']} epochs).
Best performance achieved at epoch {s['best_epoch']}.

Key Metrics:
- mAP@50: {s['best_map50']:.4f} (mean average precision at 50% IoU)
- mAP@50-95: {s['best_map5095']:.4f} (mAP across all IoU thresholds)
- Precision: {s['best_precision']:.4f} (accuracy of positive predictions)
- Recall: {s['best_recall']:.4f} (completeness of detections)

Final Loss: {s['final_val_box_loss']:.4f}
        """
        return summary.strip()
    
    def analyze_strengths(self):
        """Identify model strengths."""
        s = self.training_summary
        strengths = []
        
        if s['best_map5095'] > 0.5:
            strengths.append(f"✅ Excellent precision: mAP@50-95 = {s['best_map5095']:.4f} indicates strong overall detection accuracy")
        elif s['best_map5095'] > 0.3:
            strengths.append(f"✅ Good precision: mAP@50-95 = {s['best_map5095']:.4f}")
        
        if s['best_recall'] > 0.7:
            strengths.append(f"✅ High recall: {s['best_recall']:.4f} - model captures most objects in images")
        
        if s['best_precision'] > 0.7:
            strengths.append(f"✅ High precision: {s['best_precision']:.4f} - low false positive rate")
        
        if s['final_val_box_loss'] < 1.5:
            strengths.append(f"✅ Low validation loss: {s['final_val_box_loss']:.4f} - good convergence")
        
        return strengths if strengths else ["Model shows baseline performance"]
    
    def analyze_weaknesses(self):
        """Identify areas for improvement."""
        s = self.training_summary
        weaknesses = []
        
        if s['best_map5095'] < 0.3:
            weaknesses.append(f"⚠️  Low mAP@50-95: {s['best_map5095']:.4f} - consider more training epochs or dataset augmentation")
        
        if s['best_recall'] < 0.5:
            weaknesses.append(f"⚠️  Low recall: {s['best_recall']:.4f} - model misses many objects; try higher IoU threshold adjustments")
        
        if s['best_precision'] < 0.5:
            weaknesses.append(f"⚠️  Low precision: {s['best_precision']:.4f} - too many false positives; model may need refinement")
        
        if s['final_val_box_loss'] > 2.0:
            weaknesses.append(f"⚠️  High validation loss: {s['final_val_box_loss']:.4f} - model may be underfitting")
        
        return weaknesses if weaknesses else ["No major weaknesses detected"]
    
    def generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []
        s = self.training_summary
        
        if s['best_map5095'] < 0.4:
            recommendations.append("📈 Increase training epochs for better convergence")
            recommendations.append("🔄 Apply stronger data augmentation (rotation, brightness)")
            recommendations.append("📊 Ensure dataset quality - check for mislabeled images")
        
        if s['best_recall'] < 0.6:
            recommendations.append("🎯 Reduce confidence threshold in inference for higher recall")
            recommendations.append("📦 Check if small objects are underrepresented in training")
        
        if s['best_precision'] < 0.6:
            recommendations.append("🔍 Increase IoU threshold for post-processing NMS")
            recommendations.append("⚙️  Consider confidence threshold tuning")
        
        if recommendations == []:
            recommendations.append("✨ Model shows good performance! Monitor validation metrics in future training runs.")
        
        return recommendations
    
    def print_full_report(self):
        """Print formatted full report."""
        report = self.generate_report()
        
        print("\n" + "="*70)
        print("📋 PERFORMANCE ANALYSIS REPORT")
        print("="*70)
        
        print(f"\n📌 Model: {report['model']}")
        print(f"📅 Generated: {report['timestamp']}")
        
        print(f"\n{report['summary']}")
        
        print("\n✅ STRENGTHS:")
        for strength in report['strengths']:
            print(f"   {strength}")
        
        print("\n⚠️  WEAKNESSES:")
        for weakness in report['weaknesses']:
            print(f"   {weakness}")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*70 + "\n")
        
        return report

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("\n" + "🏀 " * 25)
    print("BASKETBALL DETECTION - MODEL PERFORMANCE METRICS")
    print("🏀 " * 25 + "\n")
    
    # Step 1: Training Analysis
    print("📊 Analyzing training history...")
    trainer = TrainingAnalyzer(MetricsConfig.RESULTS_CSV)
    training_summary = trainer.print_summary()
    
    if training_summary:
        print("📈 Generating training visualization charts...")
        trainer.plot_training_curves()
        trainer.plot_overfitting_analysis()
    
    # Step 2: Model Evaluation
    print("\n🔍 Evaluating model on test set...")
    evaluator = ModelEvaluator(MetricsConfig.MODEL_PATH, MetricsConfig.DATA_YAML)
    val_results = evaluator.validate()
    
    if val_results:
        evaluator.print_validation_report()
        print("📊 Generating validation visualization...")
        evaluator.plot_validation_results()
    
    # Step 3: Performance Insights
    print("\n💡 Generating performance insights...")
    insights = PerformanceInsights(training_summary, val_results)
    report = insights.print_full_report()
    
    # Step 4: Save Report as JSON
    if report:
        report_path = MetricsConfig.OUTPUT_DIR / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved: {report_path}")
    
    print("✨ Metrics analysis complete!")
    print(f"📁 All reports saved to: {MetricsConfig.OUTPUT_DIR.absolute()}\n")

if __name__ == "__main__":
    main()
