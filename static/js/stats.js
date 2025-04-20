const renderCharts = (categoryData, categoryLabels, spentByData, spentByLabels) => {
  // Wait for next tick to ensure DOM is ready
  setTimeout(() => {
    try {
      // Destroy existing charts if they exist
      const existingCategoryChart = Chart.getChart('categoryChart');
      if (existingCategoryChart) {
        existingCategoryChart.destroy();
      }
      
      const existingSpentByChart = Chart.getChart('spentByChart');
      if (existingSpentByChart) {
        existingSpentByChart.destroy();
      }

      // Category Chart
      const categoryCtx = document.getElementById('categoryChart');
      if (!categoryCtx) {
        console.error('Category chart canvas not found');
        return;
      }

      new Chart(categoryCtx, {
        type: 'doughnut',
        data: {
          labels: categoryLabels,
          datasets: [{
            data: categoryData,
            backgroundColor: [
              '#FF6384',
              '#36A2EB',
              '#FFCE56',
              '#4BC0C0',
              '#9966FF',
              '#FF9F40'
            ],
            hoverOffset: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
            },
            title: {
              display: true,
              text: 'Expenses by Category'
            }
          }
        }
      });

      // Spent By Chart
      const spentByCtx = document.getElementById('spentByChart');
      if (!spentByCtx) {
        console.error('Spent by chart canvas not found');
        return;
      }

      new Chart(spentByCtx, {
        type: 'doughnut',
        data: {
          labels: spentByLabels,
          datasets: [{
            data: spentByData,
            backgroundColor: [
              '#FF9F40',
              '#4BC0C0',
              '#FFCE56',
              '#36A2EB',
              '#FF6384'
            ],
            hoverOffset: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
            },
            title: {
              display: true,
              text: 'Expenses by Spender'
            }
          }
        }
      });
    } catch (error) {
      console.error('Error rendering charts:', error);
    }
  }, 0);
};

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log("Fetching chart data...");
  fetch("/expense_category_summary")
    .then((res) => res.json())
    .then((results) => {
      console.log("Chart data received:", results);
      const category_data = results.expense_category_data;
      const spent_by_data = results.expense_spent_by_data;
      
      if (!category_data || !spent_by_data) {
        throw new Error('Invalid data received from server');
      }

      const [categoryLabels, categoryValues] = [
        Object.keys(category_data),
        Object.values(category_data)
      ];
      
      const [spentByLabels, spentByValues] = [
        Object.keys(spent_by_data),
        Object.values(spent_by_data)
      ];

      renderCharts(categoryValues, categoryLabels, spentByValues, spentByLabels);
    })
    .catch(error => {
      console.error("Error fetching chart data:", error);
    });
});
