#  DDOS Attack Detection ML Model (DDOS Saldırı Tespiti ML Modeli)

--- English
> DDOS attack detection ML model was developed with feature selection algorithm. The dataset was pulled from Kaggle. (PSO integration has been made)


Particle Swarm Optimization (PSO) is an optimization algorithm inspired by the movements of bird swarms, honey bee colonies or fish schools, which are natural phenomena. The main goal of PSO is to find the best solution in a given problem.

The working principle of PSO is quite simple: A particle is randomly placed in a given space, and each particle is represented by a solution that represents a part of the problem. Each particle follows the best solution and the position of the particle that reached the best solution to update its speed and position. The particles move towards the best solution determined in this way.

One of the main advantages of PSO is that it can be quite effective and fast to solve complex optimization problems. It is also generally suitable for problems involving a large number of parameters and can be easily adapted to various types of problems.

Areas where PSO is used include engineering design, artificial neural networks, data mining, economics, and optimization of many other algorithms.

--- Türkçe

> Öznitelik seçim algoritması ile DDOS saldırı tespiti ML modeli geliştirildi. Veri seti Kaggle’dan çekildi. (PSO entegrasyonu yapıldı)

Parçacık Sürü Optimizasyonu (PSO), doğal bir olay olan kuş sürülerinin, bal arısı kolonilerinin veya balık sürülerinin hareketlerinden esinlenen bir optimizasyon algoritmasıdır. PSO'nun temel amacı, belirli bir problemdeki en iyi çözümü bulmaktır.

PSO'nun çalışma prensibi oldukça basittir: Bir parçacık, belirli bir alanda rastgele bir şekilde yerleştirilir ve her bir parçacık, problemin bir parçasını temsil eden bir çözümle temsil edilir. Her parçacık, kendi hızını ve konumunu güncellemek için en iyi çözümü ve en iyi çözüme ulaşan parçacığın konumunu takip eder. Parçacıklar, bu şekilde belirlenen en iyi çözüme doğru hareket ederler.

PSO'nun ana avantajlarından biri, karmaşık optimizasyon problemlerini çözmek için oldukça etkili ve hızlı olabilmesidir. Ayrıca, genellikle çok sayıda parametre içeren problemler için de uygundur ve çeşitli problem türlerine kolayca uyarlanabilir.

PSO'nun kullanıldığı alanlar arasında mühendislik tasarımı, yapay sinir ağları, veri madenciliği, ekonomi ve diğer birçok algoritmanın optimizasyonu bulunmaktadır.

## `Sample pso code with py`
İşte Python'da basit bir Parçacık Sürü Optimizasyonu (PSO) örneği:

```python
import numpy as np

class Particle:
    def __init__(self, dim, min_values, max_values):
        self.position = np.random.uniform(min_values, max_values, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position
        self.best_value = float('inf')

def objective_function(x):
    # Örnek amaç fonksiyonu (uyumluluk fonksiyonu)
    return np.sum(x**2)

def pso(objective_function, dim, n_particles, n_iterations, min_values, max_values):
    particles = [Particle(dim, min_values, max_values) for _ in range(n_particles)]
    global_best_position = np.random.uniform(min_values, max_values, dim)
    global_best_value = float('inf')

    for _ in range(n_iterations):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_position = particle.position
                particle.best_value = value
            if value < global_best_value:
                global_best_position = particle.position
                global_best_value = value

        for particle in particles:
            particle.velocity = 0.5 * particle.velocity + \
                                2 * np.random.rand(dim) * (particle.best_position - particle.position) + \
                                2 * np.random.rand(dim) * (global_best_position - particle.position)
            particle.position = particle.position + particle.velocity

    return global_best_position, global_best_value

# Parametreler
dim = 2  # Boyut
n_particles = 30  # Parçacık sayısı
n_iterations = 100  # İterasyon sayısı
min_values = np.array([-5, -5])  # Minimum değerler
max_values = np.array([5, 5])  # Maksimum değerler

# PSO çağrısı
best_position, best_value = pso(objective_function, dim, n_particles, n_iterations, min_values, max_values)

print("En iyi çözüm:", best_position)
print("En iyi değer:", best_value)
```

Bu kod parçacık sürüsü optimizasyonunu uygular ve en iyi çözümü ve değeri bulur. `objective_function` fonksiyonunu kendi amaç fonksiyonunuza göre değiştirebilirsiniz.
