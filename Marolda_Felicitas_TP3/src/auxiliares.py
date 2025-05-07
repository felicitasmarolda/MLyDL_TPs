import numpy as np
import models as md


def grid_search(X_train, y_train, X_val, y_val, learning_rates, adam_betas, rsl, batch_size, l2_lambda, early_stopping_patience, dropout, layer_configs):
    best_model = None
    best_val_loss = float('inf')
    best_params = None

    for lr in learning_rates:
        for beta1, beta2, epsilon in adam_betas:
            for rs in rsl:
                for bs in batch_size:
                    for l2 in l2_lambda:
                        for patience in early_stopping_patience:
                            for dp in dropout:
                                for layers in layer_configs:
                                    # Crear el modelo con los parámetros actuales
                                    model = md.MLP(layers=layers, learning_rate=lr, adam_beta1=beta1,
                                                   adam_beta2=beta2, epsilon=epsilon, random_state=rs,
                                                   batch_size=bs, l2_lambda=l2, early_stopping_patience=patience,
                                                   dropout=dp)
                                    # Entrenar el modelo
                                    model.fit(X_train, y_train)
                                    # Evaluar el modelo en el conjunto de validación
                                    val_loss = model.evaluate(X_val, y_val)

                                    # Si el modelo es mejor que el mejor encontrado hasta ahora, actualizar
                                    if val_loss < best_val_loss:
                                        best_val_loss = val_loss
                                        best_model = model
                                        best_params = {
                                            'learning_rate': lr,
                                            'adam_beta1': beta1,
                                            'adam_beta2': beta2,
                                            'epsilon': epsilon,
                                            'random_state': rs,
                                            'batch_size': bs,
                                            'l2_lambda': l2,
                                            'early_stopping_patience': patience,
                                            'dropout': dp,
                                            'layers': layers
                                        }

    return best_model, best_params

