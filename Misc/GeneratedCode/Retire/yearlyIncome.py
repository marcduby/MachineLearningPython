import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const PortfolioProjection = () => {
  const calculatePortfolioBalance = (initialBalance, withdrawalRate, realReturn, years) => {
    let balance = initialBalance;
    const data = [];
    
    for (let year = 0; year <= years; year++) {
      data.push({
        year: year + 55,
        balance: Math.round(balance),
        withdrawal: Math.round(initialBalance * withdrawalRate)
      });
      
      // Calculate next year's balance with real return of 2%
      balance = balance * (1 + realReturn) - (initialBalance * withdrawalRate);
    }
    
    return data;
  };

  const conservative = calculatePortfolioBalance(3000000, 0.03, 0.02, 45); // 3% withdrawal
  const moderate = calculatePortfolioBalance(3000000, 0.035, 0.02, 45); // 3.5% withdrawal
  const aggressive = calculatePortfolioBalance(3000000, 0.04, 0.02, 45); // 4% withdrawal

  // Combine data for chart
  const data = conservative.map((item, index) => ({
    age: item.year,
    'Conservative (3%)': item.balance,
    'Moderate (3.5%)': moderate[index].balance,
    'Aggressive (4%)': aggressive[index].balance
  }));

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>Portfolio Balance Over Time (2% Real Return)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mt-4">
          <LineChart width={800} height={400} data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="age" label={{ value: 'Age', position: 'bottom' }} />
            <YAxis 
              tickFormatter={tick => `$${(tick/1000000).toFixed(1)}M`}
              label={{ value: 'Portfolio Balance', angle: -90, position: 'left' }}
            />
            <Tooltip formatter={value => `$${(value/1000000).toFixed(2)}M`} />
            <Legend />
            <Line
              type="monotone"
              dataKey="Conservative (3%)"
              stroke="#2196F3"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="Moderate (3.5%)"
              stroke="#4CAF50"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="Aggressive (4%)"
              stroke="#FFA726"
              strokeWidth={2}
            />
          </LineChart>
        </div>
        <div className="mt-6 space-y-4">
          <p className="text-lg font-semibold">Annual Withdrawal Amounts (in today's dollars):</p>
          <ul className="list-disc pl-6 space-y-2">
            <li>Conservative (3%): ${(3000000 * 0.03).toLocaleString()} per year</li>
            <li>Moderate (3.5%): ${(3000000 * 0.035).toLocaleString()} per year</li>
            <li>Aggressive (4%): ${(3000000 * 0.04).toLocaleString()} per year</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default PortfolioProjection;