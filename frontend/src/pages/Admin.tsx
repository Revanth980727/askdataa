import React from 'react'
import { PageHeader } from '@components/ui/PageHeader'
import { Card } from '@components/ui/Card'

export const Admin: React.FC = () => {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Admin"
        description="System administration, maintenance, and health monitoring"
      />
      
      <Card>
        <div className="card-content">
          <div className="text-center py-12">
            <h3 className="text-lg font-medium text-neutral-900 mb-2">
              Admin Page
            </h3>
            <p className="text-neutral-600">
              This page will be implemented next with administration features.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
