import React from 'react'
import { PageHeader } from '@components/ui/PageHeader'
import { Card } from '@components/ui/Card'

export const Help: React.FC = () => {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Help"
        description="Documentation, tutorials, and support resources"
      />
      
      <Card>
        <div className="card-content">
          <div className="text-center py-12">
            <h3 className="text-lg font-medium text-neutral-900 mb-2">
              Help Page
            </h3>
            <p className="text-neutral-600">
              This page will be implemented next with help and documentation features.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
