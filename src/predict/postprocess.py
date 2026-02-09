import pandas as pd

def format_output(raw_prediction_data):
    """
    Transforma los datos crudos de inferencia en un DataFrame estructurado.
    Args:
        raw_prediction_data (tuple): (behavior_data, fps, class_names) procedente de run_inference.
    Returns:
        pd.DataFrame: Tabla con segundos por actividad por vaca.
    """
    behavior_data, fps, class_names = raw_prediction_data

    rows = []

    # Iterar sobre cada vaca trackeada
    for tid, actions in behavior_data.items():
        row = {'cow_id': tid}
        total_frames = 0

        # Iterar sobre cada acción detectada para esa vaca
        for cid, count in actions.items():
            action_name = class_names[cid]
            # Conversión matemática: Frames -> Segundos
            seconds = round(count / fps, 2)

            row[f"{action_name}_sec"] = seconds
            total_frames += count

        # Tiempo total que la vaca estuvo en pantalla/trackeada
        row['total_tracked_sec'] = round(total_frames / fps, 2)
        rows.append(row)

    df_pred = pd.DataFrame(rows)

    # Reordenar columnas para que cow_id y total salgan primero (estético)
    if not df_pred.empty:
        cols = ['cow_id', 'total_tracked_sec'] + [c for c in df_pred.columns if c not in ['cow_id', 'total_tracked_sec']]
        df_pred = df_pred[cols]

    return df_pred