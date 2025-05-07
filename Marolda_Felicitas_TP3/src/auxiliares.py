import numpy as np
import models as md
import metricas as mt


def grid_search(X_train, y_train, X_val, y_val, learning_rates, adam_betas, rsl, batch_size, l2_lambda, early_stopping_patience, dropout, batch_norm, layer_configs):
    best_model = None
    best_val_loss = float('inf')
    best_params = None

    for lr in learning_rates:
        for adam_p in adam_betas:
            for rs in rsl:
                for bs in batch_size:
                    for l2 in l2_lambda:
                        for patience in early_stopping_patience:
                            for dp in dropout:
                                for bn in batch_norm:
                                    for layers in layer_configs:
                                        # Crear el modelo con los parámetros actuales
                                        mejoras = {
                                            "Rate scheduling lineal": rs,
                                            "Mini batch stochastic gradient descent": bs,
                                            "L2": l2,
                                            "ADAM": adam_p,
                                            "Early stopping": patience,
                                            "Dropout": dp,
                                            "Batch normalization": bn
                                        }
                                        print(f"Entrenando modelo con parámetros: {mejoras}\nlr: {lr}\nlayers: {layers}")
                                        print()
                                        funciones_de_activacion = ['ReLU'] * (len(layers)) + ['softmax']
                                        print(f"Funciones de activación: {funciones_de_activacion}")
                                        model = md.NeuralNetwork(X_train, y_train, X_val, y_val, funciones_de_activacion, layers, mejoras, lr, 500, False)

                                        # Entrenar el modelo
                                        model.fit(X_train, y_train)
                                        print("modelo entrenado")

                                        # Evaluar el modelo en el conjunto de validación
                                        pred = model.forward_pass(X_val, False)
                                        val_loss = mt.cross_entropy(y_val, pred)
                                        print(f"Pérdida de validación: {val_loss}")

                                        # Si el modelo es mejor que el mejor encontrado hasta ahora, actualizar
                                        if val_loss < best_val_loss:
                                            best_val_loss = val_loss
                                            print(f"Nuevo mejor modelo encontrado con pérdida de validación: {val_loss}")
                                            best_model = model
                                            best_params = {
                                                'learning_rate': lr,
                                                'adam_params': adam_p,
                                                'random_state': rs,
                                                'batch_size': bs,
                                                'l2_lambda': l2,
                                                'early_stopping_patience': patience,
                                                'dropout': dp,
                                                'batch_norm': bn,
                                                'layers': layers
                                            }

    return best_model, best_params

