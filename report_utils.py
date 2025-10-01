# =========================
# Archivo: report_utils.py
# Descripción: Funciones para graficar resultados y exportar informe PDF con métricas y análisis.
# Incluye gráficas de loss/accuracy y matriz de confusión.
# Uso: Importar ReportGenerator y usar sus métodos estáticos.
# =========================

# ---- Imports ----
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- Clase ReportGenerator ----
class ReportGenerator:
    # Utilidad para graficar y exportar resultados en PDF

    @staticmethod
    def plot_loss_accuracy(history):
        # Grafica la evolución de la pérdida y la precisión por época
        plt.figure(figsize=(8,4))
        plt.plot(history['loss'], label='train_loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(history['accuracy'], label='train_acc')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='val_acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, title='Confusion matrix'):
        # Grafica la matriz de confusión
        plt.figure(figsize=(6,6))
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    @staticmethod
    def export_pdf(pdf_path, arch, acc, loss, history, cm):
        # Exporta un informe PDF con resultados, gráficas y análisis
        with PdfPages(pdf_path) as pdf:
            fig1 = plt.figure(figsize=(8.27, 11.69))
            fig1.suptitle('Informe MLP - Página 1', fontsize=14)
            plt.axis('off')
            txt = (
                "Proyecto: MLP desde cero + Intérprete -> Keras\n\n"
                "Arquitectura usada (Keras):\n" + arch + "\n\n"
                f"Resultados (Keras): test_accuracy = {acc:.4f}, test_loss = {loss:.4f}\n\n"
                "Notas:\n- Parte 1: Implementación MLP con NumPy (forward y backprop simple).\n"
                "- Parte 2: Intérprete compile_model(architecture_string) que construye un tf.keras.Sequential.\n"
                "- Parte 3: Entrenamiento y evaluación en MNIST con Keras.\n"
            )
            plt.text(0.01, 0.99, txt, fontsize=10, va='top')
            ax = fig1.add_axes([0.1, 0.15, 0.8, 0.35])
            ax.plot(history['accuracy'], label='train_acc')
            if 'val_accuracy' in history:
                ax.plot(history['val_accuracy'], label='val_acc')
            ax.set_title('Accuracy por epoch (Keras)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            pdf.savefig(fig1)
            plt.close(fig1)

            fig2 = plt.figure(figsize=(8.27, 11.69))
            fig2.suptitle('Informe MLP - Página 2', fontsize=14)
            plt.axis('off')
            analysis_text = (
                "Análisis y conclusiones\n\n"
                "1) Desafíos de implementar backprop a mano para una red grande:\n"
                "- Complejidad algorítmica y bugs numéricos (overflow/underflow).\n"
                "- Coste computacional: muchas operaciones de matriz y uso intensivo de memoria.\n"
                "- Debugging y optimizaciones (initialization, learning rate schedules, regularización).\n\n"
                "2) Ventajas de un 'compilador' (compile_model):\n"
                "- Abstracción: permite definir arquitecturas de forma legible y reproducible.\n"
                "- Reutilización: cambia la arquitectura sin tocar el código de entrenamiento.\n"
                "- Integración con frameworks: traduce a objetos optimizados y acelerados (e.g., Keras/TensorFlow).\n\n"
                "3) Recomendaciones:\n- Para mejor rendimiento usa CNNs (convolucional) en MNIST.\n- Añadir regularización (Dropout, BatchNorm) y aumentar epochs.\n"
            )
            plt.text(0.01, 0.98, analysis_text, fontsize=10, va='top')
            ax2 = fig2.add_axes([0.1, 0.05, 0.8, 0.45])
            ax2.imshow(cm, interpolation='nearest')
            ax2.set_title('Matriz de confusión (Keras MLP)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
            pdf.savefig(fig2)
            plt.close(fig2)
