class ConformalObjectDetector(nn.Module):
-> attributs: model, alpha_risk, risk_function, lambda. 
  -> def get_scores(dataloader):
     returns conformal scores. 
  -> def calibrate(calibration_dataloader):
     change la valeur de self.lambda pour calibrer le modele au niveau de risque tel que esperance de risk_function(lambda) sur la distribution de test soit au maximum alpha_risk. 
  -> def infer(dataloader):
      renvoie les prédiction sets vérifiant le niveau de risque défini par l'utilisateur (utiliser la methode get_scores)./ 