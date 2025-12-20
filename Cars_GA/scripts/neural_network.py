"""
Klasa NeuralNetwork - sieć neuronowa dla AI samochodu
"""
import numpy as np
import random


class NeuralNetwork:
    def __init__(self, config, hidden_layers=None):
        """
        Inicjalizacja sieci neuronowej
        
        Args:
            config: Słownik z konfiguracją
            hidden_layers: Opcjonalna lista warstw ukrytych (jeśli None, bierze z config)
        """
        self.config = config
        nn_config = config['neural_network']
        
        self.input_size = nn_config['input_size']
        self.output_size = nn_config['output_size']
        
        if hidden_layers is not None:
            self.hidden_layers = list(hidden_layers)
        else:
            self.hidden_layers = list(nn_config['hidden_layers'])
        
        self._build_network()

    def _build_network(self):
        """Buduje strukturę sieci na podstawie self.hidden_layers"""
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # Inicjalizacja wag i biasów
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, 
                                (self.layer_sizes[i], self.layer_sizes[i + 1]))
            b = np.zeros(self.layer_sizes[i + 1])
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, inputs):
        """
        Propagacja w przód przez sieć
        
        Args:
            inputs: Wektor wejściowy (np. odczyty czujników)
            
        Returns:
            Wektor wyjściowy (akcje)
        """
        activation = np.array(inputs)
        
        # Przez wszystkie warstwy
        for i in range(len(self.weights)):
            # Mnożenie macierzowe + bias
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            # Funkcja aktywacji
            if i < len(self.weights) - 1:
                # Warstwy ukryte: ReLU
                activation = np.maximum(0, z)
            else:
                # Warstwa wyjściowa: Sigmoid (0-1)
                activation = self._sigmoid(z)
        
        return activation
    
    def _sigmoid(self, x):
        """Funkcja sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def get_weights_flat(self):
        """
        Zwraca wszystkie wagi i biasy jako płaską tablicę
        
        Returns:
            Numpy array z wszystkimi parametrami
        """
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.extend(w.flatten())
            flat.extend(b.flatten())
        return np.array(flat)
    
    def set_weights_flat(self, flat_weights):
        """
        Ustawia wagi i biasy z płaskiej tablicy
        
        Args:
            flat_weights: Płaska tablica z parametrami
        """
        idx = 0
        for i in range(len(self.weights)):
            # Rozmiar wag
            w_size = self.weights[i].size
            w_shape = self.weights[i].shape
            
            # Wyciągnij wagi
            self.weights[i] = flat_weights[idx:idx + w_size].reshape(w_shape)
            idx += w_size
            
            # Rozmiar biasów
            b_size = self.biases[i].size
            
            # Wyciągnij biasy
            self.biases[i] = flat_weights[idx:idx + b_size]
            idx += b_size
    
    def get_num_parameters(self):
        """
        Zwraca liczbę parametrów w sieci
        
        Returns:
            Liczba parametrów
        """
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total
    
    def copy(self):
        """
        Tworzy kopię sieci
        
        Returns:
            Nowa sieć z tymi samymi wagami
        """
        new_nn = NeuralNetwork(self.config, hidden_layers=self.hidden_layers)
        
        # Kopiuj wagi i biasy
        new_nn.weights = [w.copy() for w in self.weights]
        new_nn.biases = [b.copy() for b in self.biases]
        
        return new_nn
    
    def mutate(self, mutation_rate, mutation_strength):
        """
        Mutuje wagi sieci oraz rzadko jej architekturę
        
        Args:
            mutation_rate: Prawdopodobieństwo mutacji każdej wagi
            mutation_strength: Siła mutacji
        """
        # 1. Mutacja wag i biasów
        for i in range(len(self.weights)):
            # Mutacja wag
            mask = np.random.random(self.weights[i].shape) < mutation_rate
            mutation = np.random.randn(*self.weights[i].shape) * mutation_strength
            self.weights[i] += mask * mutation
            
            # Mutacja biasów
            mask = np.random.random(self.biases[i].shape) < mutation_rate
            mutation = np.random.randn(*self.biases[i].shape) * mutation_strength
            self.biases[i] += mask * mutation
            
        # 2. Mutacja architektury (rzadka: 2% szans)
        if random.random() < 0.02:
            change_type = random.choice(['add_neuron', 'remove_neuron', 'add_layer', 'remove_layer'])
            
            if change_type == 'add_neuron' and self.hidden_layers:
                idx = random.randint(0, len(self.hidden_layers) - 1)
                self.hidden_layers[idx] = min(32, self.hidden_layers[idx] + 1)
            elif change_type == 'remove_neuron' and self.hidden_layers:
                idx = random.randint(0, len(self.hidden_layers) - 1)
                self.hidden_layers[idx] = max(1, self.hidden_layers[idx] - 1)
            elif change_type == 'add_layer':
                if len(self.hidden_layers) < 4: # Max 4 warstwy ukryte
                    self.hidden_layers.append(random.randint(4, 12))
            elif change_type == 'remove_layer':
                if len(self.hidden_layers) > 1:
                    self.hidden_layers.pop()
            
            # Przebuduj sieć (wagi zostaną zresetowane)
            self._build_network()
    
    @staticmethod
    def crossover(parent1, parent2):
        """
        Krzyżowanie dwóch sieci neuronowych
        
        Args:
            parent1: Pierwsza sieć rodzicielska
            parent2: Druga sieć rodzicielska
            
        Returns:
            Nowa sieć potomna
        """
        # Jeśli architektury są różne, weź kopię lepszego rodzica
        if parent1.hidden_layers != parent2.hidden_layers:
            return parent1.copy()
            
        child = parent1.copy()
        
        # Dla każdej warstwy
        for i in range(len(child.weights)):
            # Losowy punkt podziału dla wag
            mask = np.random.random(child.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
            
            # Losowy punkt podziału dla biasów
            mask = np.random.random(child.biases[i].shape) > 0.5
            child.biases[i] = np.where(mask, parent1.biases[i], parent2.biases[i])
        
        return child
    
    def save(self, filepath):
        """
        Zapisuje sieć do pliku
        
        Args:
            filepath: Ścieżka do pliku
        """
        data = {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath):
        """
        Wczytuje sieć z pliku
        
        Args:
            filepath: Ścieżka do pliku
            
        Returns:
            True jeśli sukces, False w przeciwnym razie
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.input_size = data['input_size']
            self.hidden_layers = data['hidden_layers']
            self.output_size = data['output_size']
            self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
            
            self.weights = [np.array(w) for w in data['weights']]
            self.biases = [np.array(b) for b in data['biases']]
            
            return True
        except Exception as e:
            print(f"Błąd wczytywania sieci: {e}")
            return False
