# AskData Frontend

A modern React application for the AskData intelligent database query system.

## 🚀 Features

- **Modern React 18** with TypeScript
- **Responsive Design** with Tailwind CSS
- **Component Library** built with Radix UI primitives
- **State Management** with Zustand
- **Routing** with React Router v6
- **Data Fetching** with React Query
- **Form Handling** with React Hook Form + Zod validation
- **Theme Support** with light/dark mode toggle
- **Accessibility** built-in with ARIA support

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **State Management**: Zustand
- **HTTP Client**: Axios
- **Data Fetching**: React Query
- **Forms**: React Hook Form + Zod
- **Icons**: Lucide React
- **Charts**: Recharts
- **Date Handling**: date-fns

## 📁 Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Base UI components (Button, Card, etc.)
│   ├── layout/         # Layout components (TopBar, Sidebar, etc.)
│   ├── connection/     # Connection-related components
│   └── providers/      # Context providers
├── pages/              # Page components
├── store/              # Zustand stores
├── hooks/              # Custom React hooks
├── utils/              # Utility functions
├── types/              # TypeScript type definitions
├── api/                # API client and services
└── styles/             # Global styles and CSS
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm 8+ or yarn 1.22+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd askdata/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

## 📝 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## 🎨 Design System

### Colors
- **Primary**: Blue shades for main actions and branding
- **Success**: Green for positive states and confirmations
- **Warning**: Yellow/Orange for caution states
- **Danger**: Red for errors and destructive actions
- **Neutral**: Gray shades for text and borders

### Components
- **Cards**: Consistent container styling with headers, content, and footers
- **Buttons**: Multiple variants (primary, secondary, outline, ghost, danger)
- **Forms**: Consistent input styling with validation states
- **Status Indicators**: Color-coded status dots and badges

## 🔧 Configuration

### Environment Variables
Create a `.env.local` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=AskData
VITE_APP_VERSION=1.0.0
```

### Tailwind Configuration
Custom colors, spacing, and animations are defined in `tailwind.config.js`.

### Vite Configuration
Path aliases and build optimizations are configured in `vite.config.ts`.

## 🏗️ Architecture

### State Management
- **Connection Store**: Manages database connections and active connection
- **Theme Store**: Manages light/dark theme preferences
- **Session Store**: Manages user session and memory (planned)

### Component Patterns
- **Compound Components**: Complex components broken into logical pieces
- **Render Props**: Flexible component composition
- **Custom Hooks**: Reusable logic extraction

### Data Flow
1. User interactions trigger actions
2. Actions update Zustand stores
3. Components react to store changes
4. API calls are made through React Query
5. UI updates reflect new state

## 📱 Responsive Design

The application is built with mobile-first responsive design:

- **Mobile**: Single column layout with collapsible sidebar
- **Tablet**: Two-column layout with expanded sidebar
- **Desktop**: Full layout with persistent sidebar

## ♿ Accessibility

- **ARIA Labels**: All interactive elements have proper labels
- **Keyboard Navigation**: Full keyboard support for all features
- **Screen Reader Support**: Semantic HTML and proper heading structure
- **High Contrast**: Theme toggle for better visibility
- **Focus Management**: Proper focus indicators and management

## 🧪 Testing

Testing setup is planned with:

- **Unit Tests**: Jest + React Testing Library
- **Integration Tests**: Component testing with user interactions
- **E2E Tests**: Playwright for full application testing

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Deploy to Static Hosting
The built application can be deployed to:
- Vercel
- Netlify
- GitHub Pages
- Any static hosting service

## 🔄 Development Workflow

1. **Feature Development**
   - Create feature branch from `main`
   - Implement feature with proper TypeScript types
   - Add tests if applicable
   - Update documentation

2. **Code Quality**
   - Run linting: `npm run lint`
   - Check types: `npm run type-check`
   - Ensure responsive design works
   - Test accessibility features

3. **Pull Request**
   - Create PR with clear description
   - Include screenshots for UI changes
   - Ensure all checks pass

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Documentation

- **Component API**: Each component has JSDoc comments
- **Type Definitions**: Comprehensive TypeScript interfaces
- **Examples**: Usage examples in component files
- **Architecture**: High-level architecture documentation

## 🐛 Troubleshooting

### Common Issues

1. **Port 3000 already in use**
   ```bash
   # Kill process using port 3000
   lsof -ti:3000 | xargs kill -9
   ```

2. **TypeScript errors**
   ```bash
   npm run type-check
   ```

3. **Build failures**
   ```bash
   # Clear node_modules and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

### Getting Help

- Check the console for error messages
- Review TypeScript type definitions
- Check component prop interfaces
- Review the backend API documentation

## 📈 Performance

### Optimization Strategies

- **Code Splitting**: Route-based code splitting with React.lazy
- **Bundle Analysis**: Analyze bundle size with build tools
- **Image Optimization**: Optimize images and use proper formats
- **Caching**: Implement proper caching strategies
- **Lazy Loading**: Load components and data on demand

### Monitoring

- **Bundle Size**: Track bundle size changes
- **Performance Metrics**: Core Web Vitals monitoring
- **Error Tracking**: Error boundary and logging
- **User Analytics**: Usage pattern analysis

## 🔮 Future Enhancements

- **PWA Support**: Service worker and offline capabilities
- **Advanced Charts**: More chart types and customization
- **Real-time Updates**: WebSocket integration for live data
- **Advanced Search**: Full-text search and filtering
- **Export Features**: PDF and Excel export capabilities
- **Multi-language**: Internationalization support

---

**Built with ❤️ by the AskData Team**
