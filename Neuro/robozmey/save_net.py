X_valid = X_valid.to(device)
y_valid = y_valid.to(device)

best_net.eval()
with torch.no_grad():
    test_preds = best_net.forward(X_valid)
    accuracy = (test_preds.argmax(dim=1) == y_valid).float().mean().data.cpu()
    print(accuracy)
    
    
torch.save(best_net.state_dict(), 'my_models/HN.pt')